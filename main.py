import PMF
import infer
import UI

model = PMF.PMF("data/ratings_train.csv")
U, V, _ = model.fit(100, learning_rate=0.0002, num_epochs=5000)
feature_joint = infer.Feature_Joint(U, V)
normal_joint = infer.Normal_Joint(U, V)
normal_joint.fit()
interface = UI.RecommendationUI("data/metadata.csv")
interface.main_loop(normal_joint, feature_joint)