import sys
import os
from src.data_generation import generate_data
from src.ocsvm_model import run_ocsvm
from src.iforest_model import run_iforest
from src.evaluation import plot_roc, write_report

def main():
    X_train, X_test, y_test = generate_data()

    ocsvm_res = run_ocsvm(X_train, X_test, y_test)
    iforest_res = run_iforest(X_train, X_test, y_test)

    plot_roc(ocsvm_res, iforest_res, y_test)
    write_report(ocsvm_res, iforest_res)

    print("Project completed successfully.")

if __name__ == "__main__":
    main()
