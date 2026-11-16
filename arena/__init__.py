"""
Adversarial Arena: Plug-and-Play Competitive Sparring Facility
Dynamic battleground for GEx and MEx policies against an adaptive Defender.
"""

from .combat_env import CombatArenaEnv, DefenderPolicy

__all__ = ["CombatArenaEnv", "DefenderPolicy"]
