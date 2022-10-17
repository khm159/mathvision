import numpy as np

class Dataset():
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
