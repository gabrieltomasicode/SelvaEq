"""
Controlador de Estado (`StateController`) para Undo/Redo e Comparação A/B (PB18 / F04 / T01 / T05).

Implementa em Python o contrato arquitetural de `StateController` previsto nos diagramas
`01_diagrama_classes.puml`, `06_sequencia_edicao_banda.puml` e `08_estados_banda_plugin.puml`.
"""

import copy
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from eq.dsp.engine import EqEngine


class StateController:
    """
    Gerencia a pilha de histórico de transações (Undo/Redo) e os slots A/B de comparação
    para o `EqEngine` sem bloquear o processamento de áudio (PB18 / F04 / T01).
    """

    def __init__(self, engine: "EqEngine", max_history: int = 64):
        self.engine = engine
        self.max_history = max_history
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self._pending_snapshot: Optional[Dict[str, Any]] = None
        self._pending_name: Optional[str] = None
        self.slot_a: Optional[Dict[str, Any]] = None
        self.slot_b: Optional[Dict[str, Any]] = None
        self.active_slot: str = "A"
        self.dirty: bool = False

    def _capture_state(self) -> Dict[str, Any]:
        """Captura um snapshot profundo e leve do estado atual das bandas e configuração."""
        bands_snapshot = tuple(self.engine.bands)
        return {
            "sample_rate": self.engine.config.sample_rate,
            "buffer_size": self.engine.config.buffer_size,
            "ms_enabled": self.engine.config.ms_enabled,
            "oversampling": self.engine.config.oversampling,
            "output_gain_db": self.engine.config.output_gain_db,
            "bypass": self.engine.config.bypass,
            "auto_gain": bool(getattr(self.engine.config, "auto_gain", False)),
            "bands": [b.to_dict() for b in bands_snapshot],
        }

    def _apply_state(self, state: Dict[str, Any]) -> None:
        """Restaura um snapshot no `EqEngine` usando a taxa efetiva e substituição atômica (T01 / T05)."""
        from eq.dsp.band import EqBand

        self.engine.config.ms_enabled = bool(state.get("ms_enabled", False))
        self.engine.config.oversampling = int(state.get("oversampling", 1))
        self.engine.config.output_gain_db = float(state.get("output_gain_db", 0.0))
        self.engine.config.bypass = bool(state.get("bypass", False))
        if "auto_gain" in state:
            self.engine.config.auto_gain = bool(state.get("auto_gain", False))

        effective_fs = int(self.engine.config.sample_rate * max(1, self.engine.config.oversampling))
        restored_bands = [
            EqBand.from_dict(b_dict, sample_rate=effective_fs)
            for b_dict in state.get("bands", [])
        ]
        self.engine.set_bands(restored_bands)

    def begin_transaction(self, name: str = "Editar EQ") -> None:
        """Inicia uma transação atômica (ex: no `mouseDown` de um nó ou slider)."""
        self._pending_name = name
        self._pending_snapshot = self._capture_state()

    def commit(self) -> None:
        """Confirma a transação iniciada (ex: no `mouseUp`), empilhando no histórico de Undo."""
        if self._pending_snapshot is not None:
            current = self._capture_state()
            if current != self._pending_snapshot:
                self._undo_stack.append({
                    "name": self._pending_name or "Transação",
                    "state": self._pending_snapshot,
                })
                if len(self._undo_stack) > self.max_history:
                    self._undo_stack.pop(0)
                self._redo_stack.clear()
                self.dirty = True
            self._pending_snapshot = None
            self._pending_name = None

    def record_Snapshot(self, name: str = "Alteração") -> None:
        """Atalho para registrar estado imediatamente antes de uma mudança pontual."""
        self.begin_transaction(name)

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> bool:
        """Desfaz a última transação e empilha o estado atual em Redo (PB18)."""
        if not self._undo_stack:
            return False
        entry = self._undo_stack.pop()
        self._redo_stack.append({
            "name": entry["name"],
            "state": self._capture_state(),
        })
        self._apply_state(entry["state"])
        self.dirty = len(self._undo_stack) > 0
        return True

    def redo(self) -> bool:
        """Refaz a última transação desfeita (PB18)."""
        if not self._redo_stack:
            return False
        entry = self._redo_stack.pop()
        self._undo_stack.append({
            "name": entry["name"],
            "state": self._capture_state(),
        })
        self._apply_state(entry["state"])
        self.dirty = True
        return True

    def store_a(self) -> None:
        """Armazena o estado atual no Slot A (PB18)."""
        self.slot_a = copy.deepcopy(self._capture_state())
        self.active_slot = "A"

    def store_b(self) -> None:
        """Armazena o estado atual no Slot B (PB18)."""
        self.slot_b = copy.deepcopy(self._capture_state())
        self.active_slot = "B"

    def compare_ab(self) -> str:
        """Alterna entre os estados armazenados nos slots A e B preservando edições do slot atual (PB18 / F04)."""
        current_snapshot = copy.deepcopy(self._capture_state())
        if self.active_slot == "A":
            self.slot_a = current_snapshot
            if self.slot_b is not None:
                self._apply_state(self.slot_b)
            else:
                self.slot_b = copy.deepcopy(current_snapshot)
            self.active_slot = "B"
        else:
            self.slot_b = current_snapshot
            if self.slot_a is not None:
                self._apply_state(self.slot_a)
            else:
                self.slot_a = copy.deepcopy(current_snapshot)
            self.active_slot = "A"
        return self.active_slot

    def copy_current_to_other_slot(self) -> str:
        """Copia o estado do slot atual para o outro slot (A->B ou B->A) (F04)."""
        snap = copy.deepcopy(self._capture_state())
        if self.active_slot == "A":
            self.slot_a = snap
            self.slot_b = copy.deepcopy(snap)
            return "B"
        self.slot_b = snap
        self.slot_a = copy.deepcopy(snap)
        return "A"

    def mark_clean(self) -> None:
        """Marca o histórico como limpo após salvar ou carregar um preset."""
        self.dirty = False
