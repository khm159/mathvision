import cv2
import os 
import numpy as np

class Dataset_face():
    def __init__(self, data_root):
        """
        input args : (str) data_root
        """
        self.root = data_root
        data_list = os.listdir(data_root)
        self.valid_ext = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
        self.data_list = []
        for data in data_list:
            if data.split(".")[1] in self.valid_ext:
                self.data_list.append(data)
        print(" {} faces founded.".format(len(self.data_list)))
        self._load_imgs()
        self.flatten()
    
    def _load_imgs(self):
        self._img_list = []
        self._data_dict = dict()
        for img in self.data_list:
            subject_id = int(img.split("s")[1].split("_")[0])
            if subject_id not in self._data_dict.keys():
                self._data_dict[subject_id] = []
            img = cv2.imread(os.path.join(self.root, img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
            self._img_list.append(img)
            self._data_dict[subject_id].append(img)
        
        self.subject_ids = list(self._data_dict.keys())
        self.subject_ids.sort()
        # for key in self.subject_ids:
        #     print("[ID] ",key, " : ", len(self._data_dict[key]))
    
    def flatten(self):
        for key in self.subject_ids:
            self._data_dict[key] = self._flatten(self._data_dict[key])
            #print(self._data_dict[key].shape)

    @staticmethod
    def _flatten(img_list):
        vector_list = [] 
        for img in img_list:
            w, h = img.shape
            vec = []
            for i in range(w):
                for j in range(h):
                    vec.append(img[i][j])
            vector_list.append(vec)
        return np.array(vector_list)
    
    @property
    def img_list(self):
        return self._vector_list

    @property
    def vec_list(self):
        return self._img_list

    @property
    def data_dict(self):
        return self._data_dict


class Dataset_apple():
    def __init__(self, data_a, data_b, test):
        """
        input args : (str) data_a 
                     path of dataset a txt file
                     (str) data_b
                     path of dataset b txt file 
                     (str) test
                     path of test dataset txt file 
        """
        self._data_a = self._process_data(data_a)
        self._data_b = self._process_data(data_b)
        self._data_test = self._process_data(test)
        self._get_data_vectors()
        # concateneated train set 
        self._data_cat_vecs = np.concatenate(
            (self._data_a_vecs, self.data_b_vecs),
            axis=0
        )

    def _get_data_vectors(self):
        """
        data to 4-dim vectors 
        """
        self._data_a_vecs = self._data_dict_to_vectors(
            self._data_a
        )
        self._data_b_vecs = self._data_dict_to_vectors(
            self._data_b
        )
        self._data_test_vecs = self._data_dict_to_vectors(
            self._data_test
        )
    
    @staticmethod
    def _data_dict_to_vectors(data_dict):
        """
        input data_dict 
        return np.array, n-dim vectors
               shape: [num_sample, num_dim]
        """
        if not isinstance(data_dict, dict):
            ValueError
        vars = list(data_dict.keys())
        num_sample = max([len(data_dict[x]) for x in vars])
        for i, var in enumerate(vars):
            data = data_dict[var]
            if not isinstance(data, list):
                ValueError
            data=np.expand_dims(data,axis=1)
            if i == 0:
                vecs = data
                continue
            vecs = np.concatenate((vecs, data), axis=1)
        return vecs

    @staticmethod
    def _process_data(data_path):
        """
        input dataset txt path 
        retun dataset information dict 
        
        dataset ordering : sugar, density, color, water
        """
        data_dict = dict()
        raw_data = [x.strip("\n") for x in open(data_path, "r").readlines()]
        variations = ['sugar', 'density', 'color', 'water']
        for var in variations:data_dict[var]=[]
        for line in raw_data:
            split = line.split(",")
            sugar, density, color, water = split
            data_dict['sugar'].append(float(sugar))
            data_dict['density'].append(float(density))
            data_dict['color'].append(float(color))
            data_dict['water'].append(float(water))
        for var in variations:data_dict[var]=np.array(data_dict[var])
        return data_dict

    @property
    def data_a(self):
        return self._data_a  

    @property
    def data_b(self):
        return self._data_b  

    @property
    def data_test(self):
        return self._data_test  

    @property
    def data_a_vecs(self):
        return self._data_a_vecs  

    @property
    def data_b_vecs(self):
        return self._data_b_vecs  

    @property
    def data_test_vecs(self):
        return self._data_test_vecs  

    @property
    def data_cat_vecs(self):
        return self._data_cat_vecs 
