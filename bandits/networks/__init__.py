__all__ = [
    "available_bandit_networks",
]

from .bandit_network import BanditNetwork, CombinatorialBanditNetwork

available_bandit_networks = {
    "BanditNetwork": BanditNetwork,
    "CombinatorialBanditNetwork": CombinatorialBanditNetwork,
}
