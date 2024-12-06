import factorize
import infer
import RecsUI

model = factorize.PMF("data/ratings_train.npy")
U, V, cov_U, cov_V, _ = model.fit(25, num_epochs=1500)
predictor = infer.Predictor(U, V, cov_U, cov_V)
interface = RecsUI.RecsUI("data/metadata.csv", predictor)
interface.main_loop()