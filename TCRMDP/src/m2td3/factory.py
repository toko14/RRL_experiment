from typing import Annotated, Dict, List

import gymnasium as gym
from rrls.envs import (
    AntParamsBound,
    HalfCheetahParamsBound,
    HopperParamsBound,
    HumanoidStandupParamsBound,
    InvertedPendulumParamsBound,
    Walker2dParamsBound,
)

ENV_NAME = {
    "Ant",
    "HalfCheetah",
    "Hopper",
    "HumanoidStandup",
    "InvertedPendulum",
    "Walker",
}

BOUNDS = {
    "Ant": AntParamsBound,
    "HalfCheetah": HalfCheetahParamsBound,
    "Hopper": HopperParamsBound,
    "HumanoidStandup": HumanoidStandupParamsBound,
    "InvertedPendulum": InvertedPendulumParamsBound,
    "Walker": Walker2dParamsBound,
}


def env_factory(env_name: str) -> gym.Env:
    # NOTE: Do not create all envs at import-time. Create only the requested one.
    ENV_IDENTIFIERS = {
        "Ant": "rrls/robust-ant-v0",
        "HalfCheetah": "rrls/robust-halfcheetah-v0",
        "Hopper": "rrls/robust-hopper-v0",
        "HumanoidStandup": "rrls/robust-humanoidstandup-v0",
        "InvertedPendulum": "rrls/robust-invertedpendulum-v0",
        "Walker": "rrls/robust-walker-v0",
    }
    env_id = ENV_IDENTIFIERS[env_name]
    return gym.make(env_id)


def bound_factory(env_name, nb_dim: int) -> Dict[str, Annotated[List[float], 2]]:
    bound = BOUNDS[env_name]

    # NOTE:
    # RRLS 側の `HalfCheetahParamsBound.ONE_DIM` は
    #   {"worldfriction": [0.1, 3.0]}
    # となっているが、本実装では 2/3 次元と同様に上限 4.0 を使いたい。
    # Enum のクラス属性を直接書き換えないように、一度 dict コピーしてから
    # HalfCheetah 1 次元のときだけ上限を 4.0 に差し替える。
    if nb_dim == 3:
        return bound.THREE_DIM.value
    if nb_dim == 2:
        return bound.TWO_DIM.value

    one_dim = dict(bound.ONE_DIM.value)
    if env_name == "HalfCheetah" and "worldfriction" in one_dim:
        low, _ = one_dim["worldfriction"]
        one_dim["worldfriction"] = [low, 4.0]
    return one_dim
