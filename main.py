import factorize
import infer
import RecsUI

model = factorize.PMF("data/ratings_train.csv")
U, V, cov_U, cov_V, _ = model.fit(100, num_epochs=1000)
predictor = infer.Predictor(U, V, cov_U, cov_V)
interface = RecsUI.RecsUI("data/metadata.csv", predictor)
interface.main_loop()