
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
def read_data():
 
    datafile = "D:/DATA MINING/Cancer dataset/data.csv"
    labels_file = "D:/DATA MINING/Cancer dataset/labels.csv"
    data = np.genfromtxt(
    datafile,
    delimiter=",",
    usecols=range(1, 20532),
    skip_header=1
    )
    true_label_names = np.genfromtxt(
    labels_file,
    delimiter=",",
    usecols=(1,),
    skip_header=1,
    dtype="str"
    )
    print(data.shape)
    print(data[:5, :3])
    print(true_label_names[:5])

    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(true_label_names)
    print(true_labels[:5])
    print(label_encoder.classes_)
    return data, true_labels