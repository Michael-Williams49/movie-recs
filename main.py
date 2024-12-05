import factorize
import infer
import RecsUI

model = factorize.PMF("data/ratings_train.csv")
U, V, _ = model.fit(100, num_epochs=1000)
feature_joint = infer.Feature_Joint(U, V)
normal_joint = infer.Normal_Joint(U, V)
normal_joint.fit()
interface = RecsUI.RecsUI("data/metadata.csv", normal_joint, feature_joint)
interface.main_loop()