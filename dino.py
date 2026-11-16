import os
import time
import random

import numpy as np
from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

# --- Гиперпараметры -----------------------------------------------------
EPISODES = 50              # количество эпизодов обучения
ALPHA = 0.1                # скорость обучения
GAMMA = 0.9                # дисконтирование
EPSILON = 0.1              # доля случайных действий при обучении
STEP_DELAY = 0.02          # пауза между шагами (сек)
MAX_STEPS = 100000         # предохранитель на один эпизод (шагов)
MAX_EPISODE_SECONDS = 120  # предохранитель на один эпизод (секунд)
MODEL_FILE = "q_table.npy"

TRAIN_MODE = True          # обучать таблицу
LOAD_MODEL = True          # загружать сохранённую таблицу
SAVE_MODEL = True          # сохранять таблицу
SAFETY_OVERRIDE = True     # гарантированно правильные действия в опасной зоне

# --- Действия -----------------------------------------------------------
ACTION_NONE = 0
ACTION_JUMP = 1
ACTION_DUCK = 2
NUM_ACTIONS = 3

# --- Состояние ----------------------------------------------------------
# reaction: тип требуемой реакции
#   0 = прыгать (кактусы и низколетящие птицы)
#   1 = приседать (птица высоко над землёй)
#   2 = препятствия нет
# bucket: дистанция до препятствия
#   0 = далеко (> DIST_ACTION), 1 = средне (DIST_NEAR..DIST_ACTION), 2 = близко
REACT_JUMP = 0
REACT_DUCK = 1
REACT_NONE = 2

DIST_ACTION = 130          # с этой дистанции начинаем действовать
DIST_NEAR = 70             # ближняя зона
PTERO_LOW_Y = 60           # птица с yPos <= этого значения требует приседа

NUM_REACTIONS = 3
NUM_BUCKETS = 3
NUM_STATES = NUM_REACTIONS * NUM_BUCKETS  # 9

# --- Чтение состояния игры (JS) ----------------------------------------
READ_STATE_JS = """
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) {
        return { status: "NO_RUNNER" };
    }
    // Игровой счёт (то, что видно на экране).
    // distanceRan хранится в пикселях, для отображения умножается на 0.025.
    const score = Math.round((runner.distanceRan || 0) * 0.025);

    if (runner.crashed) {
        return { status: "DEAD", score: score };
    }
    if (!runner.playing) {
        return { status: "NOT_PLAYING", score: score };
    }

    const dino = runner.tRex;
    const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
    const jumping = dino ? !!dino.jumping : false;

    if (obstacles.length === 0) {
        return {
            status: "OK",
            hasObstacle: false,
            distance: 1000,
            obstacleType: null,
            obstacleY: null,
            jumping: jumping,
            score: score
        };
    }

    const obstacle = obstacles[0];
    return {
        status: "OK",
        hasObstacle: true,
        distance: obstacle.xPos - dino.xPos,
        obstacleType: obstacle.typeConfig ? obstacle.typeConfig.type : null,
        obstacleY: obstacle.yPos,
        jumping: jumping,
        score: score
    };
}
"""

q_table = {s: [0.0] * NUM_ACTIONS for s in range(NUM_STATES)}


def reset_q_table():
    return {s: [0.0] * NUM_ACTIONS for s in range(NUM_STATES)}


def load_q_table():
    global q_table
    if LOAD_MODEL and os.path.exists(MODEL_FILE):
        try:
            data = np.load(MODEL_FILE, allow_pickle=True).item()
            if isinstance(data, dict) and len(data) == NUM_STATES:
                q_table = {
                    int(k): [float(v) for v in vals] for k, vals in data.items()
                }
                print("Загружена сохранённая Q-таблица.")
                return
            print("Сохранённая таблица несовместима по размеру — создаём новую.")
        except Exception as error:
            print(f"Не удалось загрузить таблицу ({error}) — создаём новую.")
    q_table = reset_q_table()
    print("Q-таблица создана с нуля.")


def save_q_table():
    if SAVE_MODEL:
        np.save(MODEL_FILE, q_table)
        print(f"Q-таблица сохранена в файл {MODEL_FILE}")


def _distance_bucket(distance):
    if distance > DIST_ACTION:
        return 0
    if distance > DIST_NEAR:
        return 1
    return 2


def _required_reaction(info):
    """Определяет требуемое действие по данным о препятствии."""
    if not info.get("hasObstacle"):
        return REACT_NONE

    obstacle_type = info.get("obstacleType")
    obstacle_y = info.get("obstacleY")

    if obstacle_type == "PTERODACTYL":
        if obstacle_y is not None and obstacle_y <= PTERO_LOW_Y:
            return REACT_DUCK
        return REACT_JUMP

    # Кактусы и прочие наземные препятствия
    return REACT_JUMP


def get_state(page):
    """Возвращает (state, info). state == -1 означает неактивную игру."""
    try:
        result = page.evaluate(READ_STATE_JS)
    except Exception as error:
        return -1, {"status": "ERROR", "message": str(error)}

    status = result.get("status")
    if status != "OK":
        return -1, {"status": status}

    reaction = _required_reaction(result)
    bucket = _distance_bucket(result["distance"])
    state = reaction * NUM_BUCKETS + bucket

    info = {
        "status": "OK",
        "hasObstacle": result["hasObstacle"],
        "distance": result["distance"],
        "obstacleType": result.get("obstacleType"),
        "obstacleY": result.get("obstacleY"),
        "jumping": result.get("jumping", False),
        "reaction": reaction,
        "bucket": bucket,
        "score": result.get("score", 0),
    }
    return state, info


def choose_action(state, info):
    """Выбирает действие: исследование, жадная политика и аварийный оверрайд."""
    if TRAIN_MODE and random.random() < EPSILON:
        action = random.choice(range(NUM_ACTIONS))
    else:
        action = int(np.argmax(q_table[state]))

    if SAFETY_OVERRIDE and info.get("status") == "OK":
        distance = info["distance"]
        reaction = info["reaction"]
        jumping = info.get("jumping", False)

        in_zone = info.get("hasObstacle") and 0 <= distance <= DIST_ACTION

        if in_zone:
            if reaction == REACT_JUMP:
                if not jumping:
                    action = ACTION_JUMP
                else:
                    action = ACTION_NONE
            elif reaction == REACT_DUCK:
                action = ACTION_DUCK
        else:
            # Вне опасной зоны нет смысла приседать или прыгать: это
            # задерживает реакцию (снятие приседа перед прыжком).
            action = ACTION_NONE

    return action


def apply_action(page, action, ducking_now):
    """Выполняет действие. Возвращает новое состояние приседа."""
    if action == ACTION_DUCK:
        if not ducking_now:
            page.keyboard.down("ArrowDown")
            return True
        return True

    # любое другое действие снимает присед
    if ducking_now:
        page.keyboard.up("ArrowDown")

    if action == ACTION_JUMP:
        page.keyboard.press("Space")

    return False


def compute_reward(prev_info, next_state, action):
    if next_state == -1:
        return -100.0
    return 1.0


def update_q_table(state, action, reward, next_state):
    current_q = q_table[state][action]

    if next_state == -1:
        max_next_q = 0.0
    else:
        max_next_q = max(q_table[next_state])

    new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
    q_table[state][action] = new_q


def wait_for_game(page, attempts=15, delay=1.5):
    """Открывает игру с ретраями, пока не появится объект Runner."""
    for attempt in range(1, attempts + 1):
        try:
            page.goto(GAME_URL, wait_until="load", timeout=30000)
        except Exception:
            time.sleep(delay)
            continue

        time.sleep(1.5)
        try:
            ready = page.evaluate("() => !!window.Runner")
        except Exception:
            ready = False

        if ready:
            return True
        time.sleep(delay)
    return False


def start_game(page):
    """Запускает/перезапускает игру. Возвращает (state, info).

    Клик по canvas блокируется оверлеем самой игры, поэтому управление
    идёт только через клавиатуру. После смерти/паузы Space перезапускает забег.
    """
    for _ in range(8):
        state, info = get_state(page)

        if state != -1:
            return state, info

        status = info.get("status")
        if status == "NO_RUNNER":
            # Страница не догрузилась — перезагружаем
            wait_for_game(page)
            continue

        # DEAD / NOT_PLAYING — Space перезапускает игру
        page.keyboard.press("Space")
        time.sleep(0.5)

    return get_state(page)


def print_q_table():
    reaction_names = {REACT_JUMP: "прыжок", REACT_DUCK: "присед", REACT_NONE: "нет"}
    bucket_names = {0: "далеко", 1: "средне", 2: "близко"}
    action_names = {ACTION_NONE: "НИЧЕГО", ACTION_JUMP: "ПРЫЖОК", ACTION_DUCK: "ПРИСЕСТЬ"}

    print("\nИтоговая Q-таблица:")
    for state in range(NUM_STATES):
        values = q_table[state]
        best = int(np.argmax(values))
        reaction = state // NUM_BUCKETS
        bucket = state % NUM_BUCKETS
        print(
            f"[{state}] реакция={reaction_names[reaction]:<7} "
            f"дистанция={bucket_names[bucket]:<7} "
            f"Q=[{values[0]:6.2f}, {values[1]:6.2f}, {values[2]:6.2f}] "
            f"-> {action_names[best]}"
        )


def run_episode(page, episode):
    state, info = start_game(page)

    if state == -1:
        print(f"Игра не запустилась (статус: {info.get('status')}). Перезапуск...")
        if not wait_for_game(page):
            print("Не удалось загрузить игру — пропускаем эпизод.")
        return 0.0, 0

    total_reward = 0.0
    ducking = False
    step = 0
    started_at = time.time()
    last_score = info.get("score", 0)

    try:
        for step in range(1, MAX_STEPS + 1):
            # Действуем сразу по свежему состоянию, без задержки между
            # чтением и нажатием клавиши (иначе бот опаздывает).
            action = choose_action(state, info)
            ducking = apply_action(page, action, ducking)

            time.sleep(STEP_DELAY)
            next_state, next_info = get_state(page)

            reward = compute_reward(info, next_state, action)

            if TRAIN_MODE:
                update_q_table(state, action, reward, next_state)

            total_reward += reward
            if next_state != -1:
                last_score = next_info.get("score", last_score)
            state, info = next_state, next_info

            if next_state == -1:
                print(
                    f"Эпизод {episode}: смерть на шаге {step}. "
                    f"Счёт игры: {last_score}. "
                    f"Награда бота: {total_reward:.0f}. "
                    f"Статус: {info.get('status')}"
                )
                break

            if time.time() - started_at > MAX_EPISODE_SECONDS:
                print(
                    f"Эпизод {episode}: выжил {MAX_EPISODE_SECONDS} с "
                    f"({step} шагов). Счёт игры: {last_score}. "
                    f"Награда бота: {total_reward:.0f}"
                )
                break
    finally:
        if ducking:
            try:
                page.keyboard.up("ArrowDown")
            except Exception:
                pass

    return total_reward, step


def main():
    print("Запускаем программу...")
    load_q_table()

    with sync_playwright() as p:
        print("Запускаем Firefox...")
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()

        try:
            print("Открываем игру...")
            if not wait_for_game(page):
                print("Не удалось открыть игру. Проверьте подключение к интернету.")
                return

            time.sleep(0.5)

            for episode in range(1, EPISODES + 1):
                print(f"\n--- Игра №{episode} ---")
                run_episode(page, episode)

                if TRAIN_MODE and episode % 5 == 0:
                    save_q_table()

            print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print_q_table()

            if TRAIN_MODE:
                save_q_table()

            input("\nНажми Enter, чтобы закрыть браузер...")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
