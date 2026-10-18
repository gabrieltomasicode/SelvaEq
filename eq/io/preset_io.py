"""
Persistência de Presets e Estado (`PresetIO`).

Refatorado a partir de `synth/file_io.py` do SelvaSonic (`save_config`, `load_config`, `enum_to_str`):
- Preserva a conversão segura de `Enum` para `str` (`enum_to_str`) e a proteção contra arrays NumPy
  não serializáveis (`last_audio_buffer`).
- Estende a serialização para salvar e restaurar tanto a `EqConfig` quanto a lista das 24 bandas
  (`EqBand`) do `EqEngine` em JSON (protótipo Python) e XML (compatível com a árvore APVTS do JUCE - PB13/PB14/PB15).
- Inclui banco de presets de fábrica (`FACTORY_PRESETS`) e aplicação com suporte a `Undo/Redo` (F10).
- Remove o `import tkinter` que existia em `synth/file_io.py`, mantendo isolamento arquitetural (PB28).
"""

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List
import json
import xml.etree.ElementTree as ET

from eq.config import EqConfig
from eq.dsp.band import EqBand
from eq.dsp.engine import EqEngine


class PresetIO:
    """
    Módulo/classe utilitária para exportação, carregamento e presets de fábrica do Equalizador (PB13, PB14, PB15, F10).
    """

    FACTORY_PRESETS: Dict[str, Dict[str, Any]] = {
        "Flat (7 Bandas Padrão)": {
            "config": {"ms_enabled": False, "oversampling": 1, "output_gain_db": 0.0, "bypass": False},
            "bands": [
                {"index": 0, "freq_hz": 30.0, "gain_db": 0.0, "q": 0.707, "ftype": "LOW_CUT", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 1, "freq_hz": 100.0, "gain_db": 0.0, "q": 0.707, "ftype": "LOW_SHELF", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 2, "freq_hz": 350.0, "gain_db": 0.0, "q": 1.0, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 3, "freq_hz": 1000.0, "gain_db": 0.0, "q": 1.0, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 4, "freq_hz": 3500.0, "gain_db": 0.0, "q": 1.0, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 5, "freq_hz": 10000.0, "gain_db": 0.0, "q": 0.707, "ftype": "HIGH_SHELF", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 6, "freq_hz": 18000.0, "gain_db": 0.0, "q": 0.707, "ftype": "HIGH_CUT", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
            ],
        },
        "Presença Vocal": {
            "config": {"ms_enabled": False, "oversampling": 1, "output_gain_db": -1.0, "bypass": False},
            "bands": [
                {"index": 0, "freq_hz": 85.0, "gain_db": 0.0, "q": 0.707, "ftype": "LOW_CUT", "slope": "S24", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 1, "freq_hz": 240.0, "gain_db": -2.5, "q": 1.4, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 2, "freq_hz": 1200.0, "gain_db": 1.5, "q": 1.2, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 3, "freq_hz": 3800.0, "gain_db": 3.5, "q": 1.1, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 4, "freq_hz": 11000.0, "gain_db": 2.5, "q": 0.707, "ftype": "HIGH_SHELF", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
            ],
        },
        "Mastering Warmth": {
            "config": {"ms_enabled": True, "oversampling": 2, "output_gain_db": -0.5, "bypass": False},
            "bands": [
                {"index": 0, "freq_hz": 25.0, "gain_db": 0.0, "q": 0.707, "ftype": "LOW_CUT", "slope": "S24", "mode": "SIDE", "bypass": False, "solo": False},
                {"index": 1, "freq_hz": 90.0, "gain_db": 1.8, "q": 0.707, "ftype": "LOW_SHELF", "slope": "S12", "mode": "MID", "bypass": False, "solo": False},
                {"index": 2, "freq_hz": 320.0, "gain_db": -1.2, "q": 1.2, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 3, "freq_hz": 2800.0, "gain_db": 1.0, "q": 1.0, "ftype": "PEAKING", "slope": "S12", "mode": "MID", "bypass": False, "solo": False},
                {"index": 4, "freq_hz": 12000.0, "gain_db": 2.0, "q": 0.707, "ftype": "HIGH_SHELF", "slope": "S12", "mode": "SIDE", "bypass": False, "solo": False},
            ],
        },
        "Sub Bass Boost": {
            "config": {"ms_enabled": False, "oversampling": 1, "output_gain_db": -2.0, "bypass": False},
            "bands": [
                {"index": 0, "freq_hz": 22.0, "gain_db": 0.0, "q": 0.707, "ftype": "LOW_CUT", "slope": "S24", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 1, "freq_hz": 55.0, "gain_db": 5.5, "q": 1.1, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 2, "freq_hz": 110.0, "gain_db": 2.5, "q": 0.707, "ftype": "LOW_SHELF", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 3, "freq_hz": 220.0, "gain_db": -3.0, "q": 1.8, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
            ],
        },
        "De-Harshness (Corte Sibilância)": {
            "config": {"ms_enabled": False, "oversampling": 2, "output_gain_db": 0.0, "bypass": False},
            "bands": [
                {"index": 0, "freq_hz": 40.0, "gain_db": 0.0, "q": 0.707, "ftype": "LOW_CUT", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 1, "freq_hz": 3200.0, "gain_db": -2.5, "q": 2.5, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 2, "freq_hz": 6500.0, "gain_db": -4.5, "q": 3.5, "ftype": "PEAKING", "slope": "S12", "mode": "STEREO", "bypass": False, "solo": False},
                {"index": 3, "freq_hz": 16500.0, "gain_db": 0.0, "q": 0.707, "ftype": "HIGH_CUT", "slope": "S24", "mode": "STEREO", "bypass": False, "solo": False},
            ],
        },
    }

    @classmethod
    def list_factory_presets(cls) -> List[str]:
        """Retorna a lista de nomes dos presets de fábrica disponíveis (F10)."""
        return list(cls.FACTORY_PRESETS.keys())

    @classmethod
    def apply_factory_preset(cls, engine: EqEngine, preset_name: str) -> EqEngine:
        """
        Aplica imediatamente um preset de fábrica ao `EqEngine` registrando transação no `StateController` (F10).
        """
        if preset_name not in cls.FACTORY_PRESETS:
            raise ValueError(f"Preset de fábrica desconhecido: {preset_name}")
        preset_data = cls.FACTORY_PRESETS[preset_name]
        sc = getattr(engine, "state_controller", None)
        started_tx = False
        if sc is not None and sc._pending_snapshot is None:
            sc.begin_transaction(f"Preset: {preset_name}")
            started_tx = True

        cfg_data = preset_data.get("config", {})
        for field_name in ("ms_enabled", "oversampling", "output_gain_db", "bypass", "auto_gain"):
            if field_name in cfg_data:
                setattr(engine.config, field_name, cfg_data[field_name])
        engine.config.validate()
        effective_fs = int(engine.config.sample_rate * max(1, engine.config.oversampling))

        bands_data = preset_data.get("bands", [])
        restored_bands = []
        for idx, b_dict in enumerate(bands_data[: engine.config.num_bands]):
            b_copy = dict(b_dict)
            b_copy["index"] = idx
            restored_bands.append(EqBand.from_dict(b_copy, sample_rate=effective_fs))

        engine.set_bands(restored_bands)
        if started_tx and sc is not None:
            sc.commit()
        return engine

    @staticmethod
    def enum_to_str(obj: Any) -> Any:
        """
        Converte recursivamente atributos `Enum` de um objeto dataclass ou dicionário para seus nomes (`str`).
        Reutilizado e aprimorado a partir de `synth/file_io.py::enum_to_str`.
        """
        if is_dataclass(obj):
            d = asdict(obj)
        elif isinstance(obj, dict):
            d = dict(obj)
        else:
            return obj.name if isinstance(obj, Enum) or hasattr(obj, "name") else obj

        for k, v in list(d.items()):
            if isinstance(v, Enum) or hasattr(v, "name"):
                d[k] = v.name
            elif isinstance(v, dict):
                d[k] = PresetIO.enum_to_str(v)
            elif isinstance(v, list):
                d[k] = [PresetIO.enum_to_str(item) for item in v]
        return d

    @staticmethod
    def to_dict(engine: EqEngine) -> Dict[str, Any]:
        """
        Converte o estado completo do `EqEngine` (`EqConfig` + `bands`) em um dicionário
        limpo de objetos transitórios (como `last_audio_buffer`).
        """
        cfg_dict = PresetIO.enum_to_str(engine.config)
        # Proteção herdada de synth/file_io.py contra arrays NumPy em memória
        cfg_dict.pop("last_audio_buffer", None)

        return {
            "plugin": "SelvaEQ",
            "version": "1.0.0",
            "config": cfg_dict,
            "bands": [band.to_dict() for band in tuple(engine.bands)],
        }

    @staticmethod
    def from_dict(engine: EqEngine, data: Dict[str, Any]) -> EqEngine:
        """
        Restaura o estado do `EqEngine` a partir de um dicionário carregado,
        validando parâmetros e recalculando coeficientes para a taxa de amostragem ativa (T01).
        """
        cfg_data = data.get("config", {})
        for field_name in (
            "sample_rate",
            "buffer_size",
            "input_device",
            "output_device",
            "num_bands",
            "ms_enabled",
            "oversampling",
            "output_gain_db",
            "bypass",
            "auto_gain",
        ):
            if field_name in cfg_data:
                setattr(engine.config, field_name, cfg_data[field_name])

        engine.config.validate()
        effective_fs = int(engine.config.sample_rate * max(1, engine.config.oversampling))

        bands_data = data.get("bands", [])
        restored_bands = []
        for idx, b_dict in enumerate(bands_data[: engine.config.num_bands]):
            b_copy = dict(b_dict)
            b_copy["index"] = idx
            restored_bands.append(EqBand.from_dict(b_copy, sample_rate=effective_fs))

        engine.set_bands(restored_bands)
        if hasattr(engine, "state_controller") and engine.state_controller is not None:
            engine.state_controller.mark_clean()
        return engine

    @staticmethod
    def save_preset(engine: EqEngine, filename: str = "eq_preset.json") -> None:
        """
        Salva o preset do `EqEngine` em disco (JSON por padrão ou XML se terminar com `.xml` - PB14).
        """
        data = PresetIO.to_dict(engine)
        if filename.lower().endswith(".xml"):
            root = ET.Element(
                "SelvaEQState",
                attrib={"plugin": str(data["plugin"]), "version": str(data["version"])},
            )
            cfg_el = ET.SubElement(root, "Config")
            for k, v in data["config"].items():
                if v is not None:
                    cfg_el.set(str(k), str(v))
            bands_el = ET.SubElement(root, "Bands")
            for b in data["bands"]:
                b_el = ET.SubElement(bands_el, "Band")
                for k, v in b.items():
                    b_el.set(str(k), str(v))
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(filename, encoding="utf-8", xml_declaration=True)
        else:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        if hasattr(engine, "state_controller") and engine.state_controller is not None:
            engine.state_controller.mark_clean()

    @staticmethod
    def load_preset(engine: EqEngine, filename: str = "eq_preset.json") -> EqEngine:
        """
        Carrega um preset JSON ou XML do sistema de arquivos e atualiza o `EqEngine` (PB14).
        """
        if filename.lower().endswith(".xml"):
            tree = ET.parse(filename)
            root = tree.getroot()
            cfg_el = root.find("Config")
            cfg_dict: Dict[str, Any] = {}
            if cfg_el is not None:
                for k, v in cfg_el.attrib.items():
                    if k in ("sample_rate", "buffer_size", "num_bands", "oversampling"):
                        cfg_dict[k] = int(v)
                    elif k in ("output_gain_db",):
                        cfg_dict[k] = float(v)
                    elif k in ("ms_enabled", "bypass", "auto_gain"):
                        cfg_dict[k] = v.lower() == "true"
                    else:
                        cfg_dict[k] = v
            bands_list = []
            bands_el = root.find("Bands")
            if bands_el is not None:
                for b_el in bands_el.findall("Band"):
                    a = b_el.attrib
                    bands_list.append({
                        "index": int(a.get("index", 0)),
                        "freq_hz": float(a.get("freq_hz", 1000.0)),
                        "gain_db": float(a.get("gain_db", 0.0)),
                        "q": float(a.get("q", 0.70710678)),
                        "ftype": a.get("ftype", "PEAKING"),
                        "slope": a.get("slope", "S12"),
                        "mode": a.get("mode", "STEREO"),
                        "bypass": a.get("bypass", "False").lower() == "true",
                        "solo": a.get("solo", "False").lower() == "true",
                    })
            return PresetIO.from_dict(engine, {"config": cfg_dict, "bands": bands_list})

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PresetIO.from_dict(engine, data)


# Funções de nível de módulo para compatibilidade direta com chamadas estilo file_io.py
enum_to_str = PresetIO.enum_to_str
to_dict = PresetIO.to_dict
save_preset = PresetIO.save_preset
load_preset = PresetIO.load_preset
apply_factory_preset = PresetIO.apply_factory_preset
