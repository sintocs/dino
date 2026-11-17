"""Dino Runner bot: Q-learning agent driven through Playwright.

Улучшенная версия исходного скрипта:
  * параметры собраны в один dataclass ``Config``;
  * логика разделена на ``DinoAgent`` (обучение) и ``GameSession`` (браузер);
  * состояние игры обогащено — 4 зоны дистанции вместо 3;
  * награда учитывает правильность реакции, а не только факт смерти;
  * ε затухает по эпизодам (меньше случайности к концу обучения);
  * ведётся CSV-журнал и сводная статистика;
  * корректная обработка Ctrl+C и освобождение клавиатуры.

Запуск:
    python dino.py                # обучение (50 эпизодов)
    python dino.py --episodes 10  # обучение, 10 эпизодов
    python dino.py --eval         # только игра с загруженной моделью
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from playwright.sync_api import Page, sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Обучение
    episodes: int = 50
    alpha: float = 0.1            # скорость обучения
    gamma: float = 0.9            # дисконтирование
    epsilon_start: float = 0.15   # стартовая доля случайных действий
    epsilon_end: float = 0.02     # минимальная доля случайных действий
    epsilon_decay: float = 0.95   # множитель затухания за эпизод

    # Тайминги и предохранители
    step_delay: float = 0.02
    max_steps: int = 100_000
    max_episode_seconds: float = 120.0

    # Файлы
    model_file: str = "q_table.npy"
    log_file: str = "training_log.csv"

    # Режимы
    train_mode: bool = True
    load_model: bool = True
    save_model: bool = True
    safety_override: bool = True
    save_every: int = 5

    # Геометрия мира (в пикселях игрового канваса)
    dist_far: float = 180.0       # граница «далеко/средне»
    dist_action: float = 130.0    # с этой дистанции начинаем действовать
    dist_near: float = 70.0       # ближняя зона
    ptero_low_y: float = 60.0     # птица ниже этой высоты требует приседа


# ---------------------------------------------------------------------------
# Действия и реакции
# ---------------------------------------------------------------------------
class Action(IntEnum):
    NONE = 0
    JUMP = 1
    DUCK = 2


class Reaction(IntEnum):
    JUMP = 0   # кактусы и низколетящие птицы
    DUCK = 1   # птица высоко над землёй
    NONE = 2   # препятствия нет


NUM_ACTIONS = len(Action)
NUM_REACTIONS = len(Reaction)
NUM_BUCKETS = 4                      # 4 зоны дистанции
NUM_STATES = NUM_REACTIONS * NUM_BUCKETS


# ---------------------------------------------------------------------------
# JS-чтение состояния игры
# ---------------------------------------------------------------------------
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
    const ducking = dino ? !!dino.ducking : false;
    const speed = runner.currentSpeed || 0;

    if (obstacles.length === 0) {
        return {
            status: "OK",
            hasObstacle: false,
            distance: 1000,
            obstacleType: null,
            obstacleY: null,
            jumping: jumping,
            ducking: ducking,
            speed: speed,
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
        ducking: ducking,
        speed: speed,
        score: score
    };
}
"""


# ---------------------------------------------------------------------------
# Q-learning агент
# ---------------------------------------------------------------------------
class DinoAgent:
    """Хранит Q-таблицу и реализует политику поведения и обучение."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.q_table: dict[int, list[float]] = self._empty_table()
        self.epsilon = cfg.epsilon_start

    # --- Q-таблица ---------------------------------------------------------
    @staticmethod
    def _empty_table() -> dict[int, list[float]]:
        return {state: [0.0] * NUM_ACTIONS for state in range(NUM_STATES)}

    def load(self) -> None:
        cfg = self.cfg
        if not (cfg.load_model and os.path.exists(cfg.model_file)):
            print("Q-таблица создана с нуля.")
            return

        try:
            data = np.load(cfg.model_file, allow_pickle=True).item()
        except Exception as error:  # noqa: BLE001
            print(f"Не удалось загрузить таблицу ({error}) — создаём новую.")
            return

        if not isinstance(data, dict) or len(data) != NUM_STATES:
            print("Сохранённая таблица несовместима по размеру — создаём новую.")
            return

        self.q_table = {
            int(state): [float(v) for v in values] for state, values in data.items()
        }
        print("Загружена сохранённая Q-таблица.")

    def save(self, path: Optional[str] = None) -> None:
        if not self.cfg.save_model:
            return
        path = path or self.cfg.model_file
        np.save(path, self.q_table)
        print(f"Q-таблица сохранена в файл {path}")

    # --- Представление состояния ------------------------------------------
    def distance_bucket(self, distance: float) -> int:
        cfg = self.cfg
        if distance > cfg.dist_far:
            return 0
        if distance > cfg.dist_action:
            return 1
        if distance > cfg.dist_near:
            return 2
        return 3

    @staticmethod
    def required_reaction(info: dict, cfg: Config) -> Reaction:
        if not info.get("hasObstacle"):
            return Reaction.NONE

        obstacle_type = info.get("obstacleType")
        obstacle_y = info.get("obstacleY")

        if obstacle_type == "PTERODACTYL":
            if obstacle_y is not None and obstacle_y <= cfg.ptero_low_y:
                return Reaction.DUCK
            return Reaction.JUMP

        # Кактусы и прочие наземные препятствия.
        return Reaction.JUMP

    def state_from_info(self, info: dict) -> int:
        reaction = int(self.required_reaction(info, self.cfg))
        bucket = self.distance_bucket(info["distance"])
        return reaction * NUM_BUCKETS + bucket

    # --- Политика ----------------------------------------------------------
    def choose_action(self, state: int, info: dict) -> Action:
        """ε-жадный выбор с аварийным оверрайдом в опасной зоне."""
        if self.cfg.train_mode and np.random.random() < self.epsilon:
            action = Action(np.random.randint(NUM_ACTIONS))
        else:
            action = Action(int(np.argmax(self.q_table[state])))

        if self.cfg.safety_override and info.get("status") == "OK":
            action = self._safety_override(action, info)
        return action

    def _safety_override(self, action: Action, info: dict) -> Action:
        distance = info["distance"]
        reaction = info["reaction"]
        in_zone = info.get("hasObstacle") and 0 <= distance <= self.cfg.dist_action

        if not in_zone:
            # Вне опасной зоны нет смысла прыгать/приседать — это задерживает
            # реакцию (в частности, снятие приседа перед прыжком).
            return Action.NONE

        if reaction == Reaction.JUMP:
            return Action.NONE if info.get("jumping") else Action.JUMP
        if reaction == Reaction.DUCK:
            return Action.DUCK
        return Action.NONE

    # --- Награда и обновление ---------------------------------------------
    def compute_reward(self, info: dict, next_state: int, action: Action) -> float:
        if next_state == -1:
            return -100.0  # смерть или потеря управления

        if not info.get("hasObstacle"):
            return 0.5     # спокойное выживание

        distance = info["distance"]
        reaction = info["reaction"]

        if not (0 <= distance <= self.cfg.dist_action):
            # Препятствие ещё далеко: поощряем бездействие, штрафуем суету.
            return 0.1 if action == Action.NONE else -1.0

        if reaction == Reaction.JUMP:
            correct = action == Action.JUMP
        elif reaction == Reaction.DUCK:
            correct = action == Action.DUCK
        else:
            correct = action == Action.NONE
        return 2.0 if correct else -5.0

    def update(self, state: int, action: Action, reward: float, next_state: int) -> None:
        current_q = self.q_table[state][int(action)]
        max_next_q = 0.0 if next_state == -1 else max(self.q_table[next_state])
        self.q_table[state][int(action)] = current_q + self.cfg.alpha * (
            reward + self.cfg.gamma * max_next_q - current_q
        )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    # --- Диагностика -------------------------------------------------------
    def print_table(self) -> None:
        reaction_names = {Reaction.JUMP: "прыжок", Reaction.DUCK: "присед", Reaction.NONE: "нет"}
        bucket_names = {0: "далеко", 1: "средне", 2: "близко", 3: "критично"}
        action_names = {Action.NONE: "НИЧЕГО", Action.JUMP: "ПРЫЖОК", Action.DUCK: "ПРИСЕСТЬ"}

        print("\nИтоговая Q-таблица:")
        for state in range(NUM_STATES):
            values = self.q_table[state]
            best = Action(int(np.argmax(values)))
            reaction = Reaction(state // NUM_BUCKETS)
            bucket = state % NUM_BUCKETS
            print(
                f"[{state:2d}] реакция={reaction_names[reaction]:<7} "
                f"дистанция={bucket_names[bucket]:<8} "
                f"Q=[{values[0]:6.2f}, {values[1]:6.2f}, {values[2]:6.2f}] "
                f"-> {action_names[best]}"
            )


# ---------------------------------------------------------------------------
# Взаимодействие с браузером и игрой
# ---------------------------------------------------------------------------
class GameSession:
    """Управляет страницей игры: запуск, чтение состояния, действия."""

    def __init__(self, page: Page, agent: DinoAgent) -> None:
        self.page = page
        self.agent = agent
        self.ducking = False

    # --- Загрузка/запуск ---------------------------------------------------
    def wait_until_ready(self, attempts: int = 15, delay: float = 1.5) -> bool:
        for _ in range(attempts):
            try:
                self.page.goto(GAME_URL, wait_until="load", timeout=30_000)
                time.sleep(1.5)
                if self.page.evaluate("() => !!window.Runner"):
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(delay)
        return False

    def read_state(self) -> tuple[int, dict]:
        """Возвращает (state, info). state == -1 — игра неактивна."""
        try:
            result = self.page.evaluate(READ_STATE_JS)
        except Exception as error:  # noqa: BLE001
            return -1, {"status": "ERROR", "message": str(error)}

        status = result.get("status")
        if status != "OK":
            return -1, {"status": status}

        info = dict(result)
        info["reaction"] = self.agent.required_reaction(info, self.agent.cfg)
        info["bucket"] = self.agent.distance_bucket(info["distance"])
        return self.agent.state_from_info(info), info

    def start_game(self) -> tuple[int, dict]:
        for _ in range(8):
            state, info = self.read_state()
            if state != -1:
                return state, info

            if info.get("status") == "NO_RUNNER":
                self.wait_until_ready()
                continue

            # DEAD / NOT_PLAYING — Space перезапускает забег.
            self.page.keyboard.press("Space")
            time.sleep(0.5)
        return self.read_state()

    # --- Действия ----------------------------------------------------------
    def apply_action(self, action: Action) -> None:
        if action == Action.DUCK:
            if not self.ducking:
                self.page.keyboard.down("ArrowDown")
                self.ducking = True
            return

        if self.ducking:
            self.page.keyboard.up("ArrowDown")
            self.ducking = False

        if action == Action.JUMP:
            self.page.keyboard.press("Space")

    def release_keys(self) -> None:
        if self.ducking:
            try:
                self.page.keyboard.up("ArrowDown")
            except Exception:  # noqa: BLE001
                pass
            self.ducking = False

    # --- Эпизод ------------------------------------------------------------
    def run_episode(self, episode: int) -> dict:
        state, info = self.start_game()
        if state == -1:
            print(f"Игра не запустилась (статус: {info.get('status')}). Перезапуск...")
            self.wait_until_ready()
            return {"episode": episode, "steps": 0, "score": 0, "reward": 0.0, "status": "FAILED"}

        agent = self.agent
        total_reward = 0.0
        last_score = info.get("score", 0)
        step = 0
        started_at = time.time()
        status = "OK"

        self.ducking = False
        try:
            for step in range(1, agent.cfg.max_steps + 1):
                action = agent.choose_action(state, info)
                self.apply_action(action)

                time.sleep(agent.cfg.step_delay)
                next_state, next_info = self.read_state()

                reward = agent.compute_reward(info, next_state, action)
                if agent.cfg.train_mode:
                    agent.update(state, action, reward, next_state)

                total_reward += reward
                if next_state != -1:
                    last_score = next_info.get("score", last_score)
                state, info = next_state, next_info

                if next_state == -1:
                    status = info.get("status", "DEAD")
                    break

                if time.time() - started_at > agent.cfg.max_episode_seconds:
                    status = "TIME_LIMIT"
                    break
        except KeyboardInterrupt:
            status = "INTERRUPTED"
            raise
        finally:
            self.release_keys()

        result = {
            "episode": episode,
            "steps": step,
            "score": last_score,
            "reward": round(total_reward, 1),
            "status": status,
        }
        print(
            f"Эпизод {episode}: {status} на шаге {step}. "
            f"Счёт: {last_score}. Награда: {total_reward:.0f}. "
            f"ε={agent.epsilon:.3f}"
        )
        return result


# ---------------------------------------------------------------------------
# Журнал обучения
# ---------------------------------------------------------------------------
class TrainingLog:
    HEADERS = ["episode", "steps", "score", "reward", "status"]

    def __init__(self, path: str) -> None:
        self.path = path
        is_new = not os.path.exists(path)
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.HEADERS)
        if is_new:
            self._writer.writeheader()

    def write(self, row: dict) -> None:
        self._writer.writerow({key: row.get(key, "") for key in self.HEADERS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dino Runner Q-learning bot")
    parser.add_argument("--episodes", type=int, default=None, help="количество эпизодов")
    parser.add_argument("--eval", action="store_true", help="только игра без обучения")
    parser.add_argument("--headless", action="store_true", help="запустить браузер без окна")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.eval:
        cfg.train_mode = False
        cfg.save_model = False
        cfg.episodes = cfg.episodes if args.episodes else 1
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    print("Запускаем программу...")
    agent = DinoAgent(cfg)
    agent.load()

    log = TrainingLog(cfg.log_file)
    scores: list[int] = []

    try:
        with sync_playwright() as playwright:
            print("Запускаем Firefox...")
            browser = playwright.firefox.launch(headless=args.headless)
            page = browser.new_page()
            session = GameSession(page, agent)

            try:
                print("Открываем игру...")
                if not session.wait_until_ready():
                    print("Не удалось открыть игру. Проверьте подключение к интернету.")
                    return

                time.sleep(0.5)

                for episode in range(1, cfg.episodes + 1):
                    print(f"\n--- Игра №{episode} ---")
                    result = session.run_episode(episode)
                    log.write(result)
                    scores.append(result["score"])

                    if cfg.train_mode:
                        agent.decay_epsilon()
                        if episode % cfg.save_every == 0:
                            agent.save()

                print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
                if scores:
                    print(
                        f"Лучший счёт: {max(scores)} | "
                        f"Средний: {sum(scores) / len(scores):.1f} | "
                        f"Эпизодов: {len(scores)}"
                    )
                agent.print_table()

                if cfg.train_mode:
                    agent.save()

                input("\nНажми Enter, чтобы закрыть браузер...")
            finally:
                browser.close()
    except KeyboardInterrupt:
        print("\nПрервано пользователем. Сохраняем модель...")
        agent.save()
    finally:
        log.close()


if __name__ == "__main__":
    main()
