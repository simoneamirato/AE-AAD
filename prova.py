import AE_AAD
import create_dataset_unsupervisied

x_training,oracle,_ = create_dataset_unsupervisied.one_vs_all(0, 10)

AE_AAD.launch(x_training,oracle,10,2,EP=20,verb=1)