from gym.envs.registration import register
from sklearn.model_selection import train_test_split

from apievasion_rl.envs.utils import interface

# create a holdout set
sha256 = interface.get_available_sha256()
sha256.sort()
sha256_train, sha256_holdout = train_test_split(sha256, test_size=1000, random_state=10, shuffle=False)

MAXTURNS = 10

register(
    id="APIdetector-train-v0",
    entry_point="apievasion_rl.envs:APIdetectorEnv",
    kwargs={
        "random_sample": True,
        "maxturns": MAXTURNS,
        "sha256list": sha256_train,
    },
)

register(
    id="APIdetector-test-v0",
    entry_point="apievasion_rl.envs:APIdetectorEnv",
    kwargs={
        "random_sample": False,
        "maxturns": MAXTURNS,
        "sha256list": sha256_holdout,
    },
)


