"""Bottle inventory management with persistent state.

Tracks which rack slots still hold a bottle and keeps a log of every bottle
already handed out (when, from which slot, to a person at which position).
State is saved to a JSON file after every change, so the robot remembers
served bottles across restarts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

from xlerobot import config

log = logging.getLogger(__name__)


@dataclass
class ServeRecord:
    """One bottle handed out."""
    slot_id: str
    timestamp: float
    person_bearing_deg: float | None = None
    person_distance_m: float | None = None
    confidence: str | None = None


@dataclass
class _SlotState:
    slot: config.BottleSlot
    occupied: bool = True


@dataclass
class _PersistedState:
    occupied: dict[str, bool] = field(default_factory=dict)
    served: list[dict] = field(default_factory=list)


class BottleManager:
    """Knows which slots are full and remembers everything already served."""

    def __init__(self,
                 slots: list[config.BottleSlot] | None = None,
                 state_file: str = config.BOTTLE_STATE_FILE):
        slots = slots if slots is not None else config.BOTTLE_SLOTS
        self._state_file = state_file
        self._slots: dict[str, _SlotState] = {
            s.slot_id: _SlotState(slot=s) for s in slots}
        self._served: list[ServeRecord] = []
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            log.exception("Could not read %s, starting fresh", self._state_file)
            return
        for slot_id, occupied in raw.get("occupied", {}).items():
            if slot_id in self._slots:
                self._slots[slot_id].occupied = bool(occupied)
        self._served = [ServeRecord(**rec) for rec in raw.get("served", [])]
        log.info("Loaded bottle state: %d/%d slots full, %d served so far",
                 self.remaining(), len(self._slots), len(self._served))

    def _save(self):
        state = _PersistedState(
            occupied={sid: st.occupied for sid, st in self._slots.items()},
            served=[asdict(rec) for rec in self._served],
        )
        tmp = self._state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(state), fh, indent=2)
        os.replace(tmp, self._state_file)

    # -- inventory -----------------------------------------------------------

    def remaining(self) -> int:
        return sum(1 for st in self._slots.values() if st.occupied)

    def served_count(self) -> int:
        return len(self._served)

    def served_history(self) -> list[ServeRecord]:
        return list(self._served)

    def next_slot(self, preferred_arm: str | None = None) -> config.BottleSlot | None:
        """Next full slot to serve from.

        If ``preferred_arm`` is given (e.g. the arm on the same side as the
        person), slots served by that arm are tried first.
        """
        full = [st.slot for st in self._slots.values() if st.occupied]
        if not full:
            return None
        if preferred_arm is not None:
            preferred = [s for s in full if s.arm == preferred_arm]
            if preferred:
                full = preferred
        return sorted(full, key=lambda s: s.slot_id)[0]

    def mark_served(self, slot_id: str,
                    person_bearing_deg: float | None = None,
                    person_distance_m: float | None = None,
                    confidence: str | None = None):
        """Records that the bottle in ``slot_id`` was handed out."""
        if slot_id not in self._slots:
            raise KeyError(f"unknown slot {slot_id!r}")
        self._slots[slot_id].occupied = False
        self._served.append(ServeRecord(
            slot_id=slot_id,
            timestamp=time.time(),
            person_bearing_deg=person_bearing_deg,
            person_distance_m=person_distance_m,
            confidence=confidence,
        ))
        self._save()
        log.info("Served bottle from slot %s (%d left, %d served total)",
                 slot_id, self.remaining(), self.served_count())

    def mark_refilled(self, slot_id: str):
        """Marks a single slot as holding a fresh bottle again."""
        if slot_id not in self._slots:
            raise KeyError(f"unknown slot {slot_id!r}")
        self._slots[slot_id].occupied = True
        self._save()

    def refill_all(self):
        """Marks every slot full (served history is kept)."""
        for st in self._slots.values():
            st.occupied = True
        self._save()
        log.info("All %d slots refilled", len(self._slots))
