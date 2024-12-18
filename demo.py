import factorize
import infer
import RecsUI
import test

model = factorize.PMF("data/ratings_train.npy")

print("== Model Validation ==")
model.validate(0.2, 25, num_epochs=2000)

print("== Model Fit ==")
U, V, cov_U, cov_V, _ = model.fit(25, num_epochs=1500, verbose=True)

predictor = infer.Predictor(U, V, cov_U, cov_V)
tester = test.Tester("data/ratings_test.npy")

print("== Precision Test ==")
total, precision, MAE = tester.test_precision(predictor, verbose_step=1)
print(f"Total Ratings: {total}, Precision: {(precision * 100):.2f}%, MAE: {MAE:.4g}")

print("== Recall Test ==")
total, recall = tester.test_recall(predictor, verbose_step=1)
print(f"Total Ratings: {total}, Recall: {(recall * 100):.2f}%")

interface = RecsUI.RecsUI("data/metadata.csv", predictor)
interface.main_loop()