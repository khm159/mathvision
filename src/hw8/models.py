import os 
import cv2
import gc
import numpy as np
import pickle

from dataset import Dataset_apple, Dataset_face

class FaceRecognition(object):
    def __init__(self, face_root, target_data, n_component=10,pre_extract_pca_dict=None):
        self.data = Dataset_face(face_root)
        self.n_component=n_component
        self.pca_data_dict  = None
        if pre_extract_pca_dict is not None:
            with open('pca_data_dict.pkl','rb') as f:
                self.pca_data_dict = pickle.load(f)

    def __call__(self):
        print("Face analysis!")
        if self.pca_data_dict is None:
            self.faces_PCA()
        print(self.pca_data_didct.keys())

    def faces_PCA(self):
        data_dict = self.data.data_dict
        self.pca_data_dict = dict()
        for key in self.data.subject_ids:
            if key not in self.pca_data_dict.keys():
                self.pca_data_dict[key] = dict()
            data, P, D, PT  = self._PCA(
                data_dict[key]
            )
            # 모수 추정 
            mean_face, Cov_face = self.MVN_parameterization(data)
            self.pca_data_dict[key]['eigenvalues'] = D
            self.pca_data_dict[key]['eigenvectors'] = P
            self.pca_data_dict[key]['mean'] = P
            self.pca_data_dict[key]['cov'] = P

        with open('pca_data_dict.pkl','wb') as f:
            pickle.dump(self.pca_data_dict,f)
        
    def _PCA(self, data):
        """
        input  : objection vectors 
        output : principal componments  
        """
        data = self._whitening(data, show_statictics=False)
        cov_matrix = np.cov(data.T)
        gc.collect()
        eigs = np.linalg.eig(cov_matrix)
        self.eigenvalues = eigs[0]
        eigenvectors = eigs[1]
        P = eigenvectors
        D = np.diag(self.eigenvalues)
        PT = P.T
        # P^T*X
        new_coordinate = self.new_coordinates(data, P)
        index = self.eigenvalues.argsort()[::-1]
        index = list(index)
        for i in range(self.n_component):
            if i==0:
                new = [new_coordinate[:, index.index(i)]]
            else:
                new = np.concatenate(([new, [new_coordinate[:, index.index(i)]]]), axis=0)
        return new.T, P, D, PT 
    
    @staticmethod
    def proj_eigenspace(self, data, tgt_eigenvector):
        """
        projection to eigenspace
        """
        # projection to calculated eigenspace 
        data = self._whitening(data)
        new_coordinate = self.new_coordinates(data, tgt_eigenvector)
        new_coordinate = new_coordinate[:,:self.n_componment]
        return new_coordinate

    @staticmethod
    def new_coordinates(data, eigenvectors):
        """
        projection to eigenspace
        """
        for i in range(eigenvectors.shape[0]):
            if i == 0:
                new = [data.dot(eigenvectors.T[i])]
            else:
                new = np.concatenate((new, [data.dot(eigenvectors.T[i])]), axis=0)
        return new.T

    @staticmethod
    def MVN_parameterization(data):
        """
        parameterization multi-variate normal distribution 
        from the input vectors 
        """
        data = data.T
        num_dim = data.shape[1]
        num_observation = data.shape[0]
        mean_vec = np.mean(data, axis=1)
        data = data - np.expand_dims(mean_vec,axis=1)
        data_T = data.T
        C = np.matmul(data, data.T)/(num_observation-1)
        return mean_vec, C

    @staticmethod
    def _whitening(data, show_statictics=False):
        """
        z-score standalization 
        input : np.array vecs 
                shape = [num_sample, num_dim]
        return : np.array scaled_vecs
                shape = [num_sample, num_dim]
        """
        means = np.mean(data, axis=0, keepdims=True)
        std   = np.std(data, axis=0, keepdims=True)
        zero_meaned = data-means
        z_standalized = zero_meaned/std
        if show_statictics:
            print("mean")
            print(np.mean(z_standalized, axis=0, keepdims=True))
            print("var")
            print(np.var(z_standalized, axis=0, keepdims=True))
            print("std")
            print(np.std(z_standalized, axis=0, keepdims=True))
        return z_standalized

    @staticmethod
    def new_coordinates(data, eigenvectors):
        """
        projection to eigenspace
        """
        for i in range(eigenvectors.shape[0]):
            if i == 0:
                new = [data.dot(eigenvectors.T[i])]
            else:
                new = np.concatenate((new, [data.dot(eigenvectors.T[i])]), axis=0)
        return new.T

class AppleClassification(object):
    def __init__(self, data_a, data_b, test):
        self.data = Dataset_apple(data_a, data_b, test)

    def _PCA(self, data, n_componment=2):
        """
        input  : objection vectors 
        output : principal componments  
        """
        print("   >  Data scailing ")
        data = self._whitening(data, show_statictics=False)

        print("   >  Get Covariance matrix")
        cov_matrix = np.cov(data.T)

        print("   >  Get Eigen values and vectors")
        eigs = np.linalg.eig(cov_matrix)
        self.eigenvalues = eigs[0]
        eigenvectors = eigs[1]

        print("   >  Eigen decomposition")
        self.P = eigenvectors
        self.D = np.diag(self.eigenvalues)
        self.PT = self.P.T
        
        # checking. 
        print("\n   > Checking Reconstructed Covariance Matrix\n")

        # Cov = P*D*P^T
        recon_cov = np.dot(np.dot(self.P,self.D),self.PT)
        print(recon_cov)

        print("\n   > Checking Original Covariance Matrix\n")
        print(cov_matrix)
        print()
        
        print("   >  Get principal components")
        print("      Top {} P.C.".format(n_componment))
        print("      ", self.eigenvalues[:n_componment])

        # =====================[sol.1]==============================
        
        print("   >  Projection to eigenspace\n")
        # P^T*X
        new_coordinate = self.new_coordinates(data, self.P)

        index = self.eigenvalues.argsort()[::-1]
        index = list(index)
        for i in range(n_componment):
            if i==0:
                new = [new_coordinate[:, index.index(i)]]
            else:
                new = np.concatenate(([new, [new_coordinate[:, index.index(i)]]]), axis=0)
        
        self.n_componment = n_componment
        # =====================[sol.2]==============================
        return new.T
    
    def proj_eigenspace(self, data):
        """
        projection to eigenspace
        """
        # projection to calculated eigenspace 
        data = self._whitening(data)
        new_coordinate = self.new_coordinates(data, self.P)
        new_coordinate = new_coordinate[:,:self.n_componment]
        return new_coordinate

    def __call__(self):
        print("   1. Dataset Distributions")
        data_a = self.data.data_a_vecs
        data_b = self.data.data_b_vecs
        data_cat = self.data.data_cat_vecs
        test_set = self.data.data_test_vecs
        print("- The A Apple            : ", data_a.shape)
        print("- The B Apple            : ", data_b.shape)
        print("- The Apple(A+B)         : ", data_cat.shape)
        print("- Test Apple             : ", test_set.shape)
        print("- Variable dim           : ", data_a.shape[1])

        print("\n   2. Perform PCA toward all data")
        reductioned_data = self._PCA(data_cat)
        #reductioned_data = self._PCA_lib(data_cat) 
        print("- Reductioned Apple : ", reductioned_data.shape)

        print("\n   3. Visualization")
        reductioned_data_a = reductioned_data[:data_a.shape[0]]
        reductioned_data_b = reductioned_data[data_a.shape[0]:]
        print("recon a", reductioned_data_a.shape)
        print("recon b", reductioned_data_b.shape)
        self.visualization(reductioned_data_a, reductioned_data_b)

        print("\n    3. 2D-gaussian modeling")
        print("Apple A parameter ")
        mean_a, C_a = self.MVN_parameterization(reductioned_data_a)
        print("Apple B parameter ")
        mean_b, C_b = self.MVN_parameterization(reductioned_data_b)

        print("\n    4. Testset Classification")
        # 1. projection to subspace 
        test_set = self.proj_eigenspace(test_set)
        self.visualization2(
            reductioned_data_a, 
            reductioned_data_b, 
            test_set
        )
        distances = [] 
        for data in test_set:
            diss = []
            dis_a = self.calculate_distance_mahalanobis(
                mean_a, C_a, data
            )
            diss.append(dis_a)
            dis_b = self.calculate_distance_mahalanobis(
                mean_b, C_b, data
            )
            diss.append(dis_b)
            distances.append(diss)
        
        label = ['A', 'B']
        for elem in distances:
            pred = np.argmin(np.array(elem))
            print(label[pred])
            
    def calculate_distance_mahalanobis(self, mean, cov, input_vec):
        """
        Sigma = cov 
        
        rout( (x - mean)^T * inv_cov * (x - mean)  )
        input must be 1d vector
        """
        inv_cov = np.linalg.inv(cov)

        # X - mean (추정된 모수)
        input_vec = input_vec - mean
        input_vec = np.expand_dims(input_vec, axis=0)

        # cov^-1 * (x-man)
        l = np.dot(input_vec, inv_cov)

        # 2-dim gaussian mahalanobis distance 
        # rout( (x - mean)^T * inv_cov * (x - mean)  )
        dist = np.sqrt(np.dot(l, input_vec.T).diagonal())
        return dist
        
    def MVN_parameterization(self,data):
        """
        parameterization multi-variate normal distribution 
        from the input vectors 
        """
        data = data.T
        num_dim = data.shape[1]
        num_observation = data.shape[0]
        mean_vec = np.mean(data, axis=1)
        print(" Mean Vectors")
        print(mean_vec)
        data = data - np.expand_dims(mean_vec,axis=1)
        data_T = data.T
        C = np.matmul(data, data.T)/(num_observation-1)
        print(" Cov Matrix")
        print(C)
        self.twoD_gaussian_heamat(mean_vec, C)
        return mean_vec, C
        
    @staticmethod
    def visualization(a, b):
        import matplotlib.pyplot as plt
        plt.scatter(a[:,0], a[:,1], s=10, c='red')
        plt.scatter(b[:,0], b[:,1], s=10, c='blue')
        plt.show()
    
    @staticmethod
    def visualization2(a, b, test):
        import matplotlib.pyplot as plt
        plt.scatter(a[:,0], a[:,1], s=10, c='red')
        plt.scatter(b[:,0], b[:,1], s=10, c='blue')
        test_apple1 = test[0]
        test_apple2 = test[1]
        plt.scatter(test_apple1[0], test_apple1[1], s=50, c='green')
        plt.scatter(test_apple2[0], test_apple2[1], s=50, c='magenta')
        plt.show()

    @staticmethod 
    def twoD_gaussian_heamat(mean,cov):
        """
        다변수 정규분포는 2개의 모수를 가지고 있음 
        추정해야할 모수는 2개인데, 우리는 이미 이를 추정했음
        1. mean 
        2. covariance matrix 
        """
        import matplotlib.pyplot as plt
        from scipy.stats import multivariate_normal
        s1 = np.eye(2) # we are standalize the original data...so.. 

        kernal = multivariate_normal(mean=mean, cov=cov)
        # create a grid of (x,y) coordinates at which to evaluate the kernels
        xlim = (-50, 50)
        ylim = (-50, 50)
        xres = 1000
        yres = 1000

        x = np.linspace(xlim[0], xlim[1], xres)
        y = np.linspace(ylim[0], ylim[1], yres)
        xx, yy = np.meshgrid(x,y)

        # evaluate kernels at grid points
        xxyy = np.c_[xx.ravel(), yy.ravel()]
        zz = kernal.pdf(xxyy)# + k2.pdf(xxyy)

        # reshape and plot image
        img = zz.reshape((xres,yres))
        plt.imshow(img)
        plt.show()


    @staticmethod
    def _whitening(data, show_statictics=False):
        """
        z-score standalization 
        input : np.array vecs 
                shape = [num_sample, num_dim]
        return : np.array scaled_vecs
                shape = [num_sample, num_dim]
        """
        means = np.mean(data, axis=0, keepdims=True)
        std   = np.std(data, axis=0, keepdims=True)
        zero_meaned = data-means
        z_standalized = zero_meaned/std
        if show_statictics:
            print("mean")
            print(np.mean(z_standalized, axis=0, keepdims=True))
            print("var")
            print(np.var(z_standalized, axis=0, keepdims=True))
            print("std")
            print(np.std(z_standalized, axis=0, keepdims=True))
        return z_standalized

    @staticmethod
    def _PCA_lib(data, n_componment=2, show_eigenvalues=False):
        """
        ONLY USE for checking (모수 저장안함)
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(data)
        X_pca = pca.transform(data)
        return X_pca

    @staticmethod
    def new_coordinates(data, eigenvectors):
        """
        projection to eigenspace
        """
        for i in range(eigenvectors.shape[0]):
            if i == 0:
                new = [data.dot(eigenvectors.T[i])]
            else:
                new = np.concatenate((new, [data.dot(eigenvectors.T[i])]), axis=0)
        return new.T