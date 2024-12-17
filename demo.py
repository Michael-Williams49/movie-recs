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

print("== Accuracy Test ==")
total, accuracy, MAE, average_intersect = tester.test_accuracy(predictor, verbose_step=1)
print(f"Total Ratings: {total}, Average Size of Intersect: {average_intersect:.2f}, Accuracy: {(accuracy * 100):.2f}%, MAE: {MAE:.4g}")

print("== Coverage Test ==")
total, coverage = tester.test_coverage(predictor, verbose_step=1)
print(f"Total Ratings: {total}, Coverage: {(coverage * 100):.2f}%")

interface = RecsUI.RecsUI("data/metadata.csv", predictor)
interface.main_loop()