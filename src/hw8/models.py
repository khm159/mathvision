import os 
import cv2
import gc
import copy
import numpy as np
import pickle
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from dataset import Dataset_apple, Dataset_face
matplotlib_axes_logger.setLevel('ERROR')

class FaceRecognition(object):
    def __init__(self, face_root, target_data, n_component=10,pre_extract_pca_dict=None):
        self.data = Dataset_face(face_root)
        self.n_component=n_component
        self.pca_data_dict  = None
        if pre_extract_pca_dict is not None:
            with open('pca_data_dict.pkl','rb') as f:
                self.pca_data_dict = pickle.load(f)
            self.eigenvalues = self.pca_data_dict['eigenvalues']

    def MVN_parameterization(self,data):
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

    def __call__(self):
        # 0. extract features 
        print("Face analysis!")
        if self.pca_data_dict is None:
            self.faces_PCA_train()
            self.eigenvalues=self.pca_data_dict['eigenvalues']
            
        # 1. visualization 
        # self.visualize_faces2d()

        # 2. one person visualization 
        # k_list = [1, 10, 100, 200]
        # self.visualize_eigenface(
        #     k_list
        # )

        # 3. face recognition 
        k_list = [1, 10, 100, 200]
        self.recognize_id(
            k_list
        )

    def recognize_id(self, k_list):
        from sklearn.decomposition import PCA
        test_dict = self.data.test
        prefix = 'eigen_{:03d}.jpg'
        metric = "l2"

        test_subjects = list(test_dict.keys())
        test_subjects.sort()

        for i, key in enumerate(self.data.train.keys()):
            if i ==0:
                train_cat = self.data.train[key]
            else:
                train_cat = np.concatenate((train_cat, self.data.train[key]))

        save_root = "eigenfaces_test"
        if not os.path.isdir(save_root):
            os.mkdir(save_root)

        person_num_per_id = 9

        for k in k_list:
            print(" K ", k)
            #train 
            pca = PCA(
                    n_components=k, 
                    whiten=True
                )
            pca.fit(train_cat)
            ids_mean = []
            ids_cov = []
            train_transforms = pca.transform(train_cat)
            eigenvector = pca.components_

            # for each ids
            for i, key in enumerate(self.data.train.keys()):
                id_data = train_transforms[i*(person_num_per_id):(i+1)*(person_num_per_id),:]

                # parameter estimation 
                mean_vec, C = self.MVN_parameterization(id_data)
                ids_mean.append(mean_vec)
                ids_cov.append(C)
    
            Y_pred = []
            for i, sub_id in enumerate(test_subjects):
                # prepare labels 
                if i == 0:
                    Y = [sub_id]
                else:
                    Y.append(sub_id)
                
                test = self.data.test[sub_id]

                test_transformed = pca.transform(test)
                reconstructed = pca.inverse_transform(test_transformed)

                num_sample = reconstructed.shape[0]
                reconstructed = np.reshape(reconstructed, (num_sample, self.data.w, self.data.h, 1))

                # reconstruct faces using tarining db's parameters 
                for face in reconstructed:
                    cv2.imwrite(
                        os.path.join(save_root, "ID[{}]_k[{}].png".format(sub_id, k)), 
                        face
                    )
                
                # predict ids 
                # get mahalanobis distance and predict minimun distance 
                distances = []
                for i in range(len(ids_mean)):
                    lbl_mean = ids_mean[i] # (2576)
                    lbl_cov = ids_cov[i]   # (2576, 2576)
                    if metric =="l2":
                        diss = distance.euclidean(
                            test_transformed[0],
                            lbl_mean
                        )
                    elif metric =="mahalanobis":
                        diss = distance.mahalanobis(
                            test_transformed[0],
                            lbl_mean,
                            lbl_cov
                        )

                    distances.append(diss)
                lowest = 1000
                argmin = 0
                for i, elem in enumerate(distances):
                    elem = abs(elem)
                    if elem < lowest:
                        argmin = i
                        lowest = elem
                Y_pred.append(argmin+1)
            print()
            print(Y_pred)
            print(Y)
            print()
            accuracy = self.calculate_accuracy(Y, Y_pred)
            print("Accuracy : ", accuracy)

    def visualize_dist(self):
        from sklearn.decomposition import PCA
        data_dict = self.data.train
        colors = cm.rainbow(np.linspace(0, 1, len(self.data.subject_ids)))
        self.pca_data_dict = dict()

        for i, key in enumerate(self.data.train.keys()):
            if i ==0:
                train_cat = self.data.train[key]
            else:
                train_cat = np.concatenate((train_cat, self.data.train[key]))

        pca = PCA(
            n_components=2, 
            whiten=True
        )
        pca.fit(train_cat)
        proj = pca.transform(train_cat)
        num_train_per_id = 9
        for i, subject in enumerate(self.data.subject_ids):
            for j in range(num_train_per_id):
                # 1. proj 2-dim space
                proj_face_vec = proj[i*num_train_per_id+j]
                plt.scatter(
                    proj_face_vec[0],
                    proj_face_vec[1], 
                    c = colors[i],
                    s = 10
                )
        plt.show()


    def calculate_accuracy(self,Y, Y_pred):
        len_sample = len(Y)
        true = 0
        false = 0
        for i, pred in enumerate(Y_pred):
            if pred == Y[i]:
                true +=1
            else:
                false +=1
        return (true)/len_sample
                        
                        
    def calculate_distance_mahalanobis(self, mean, cov, input_vec):
        """
        Sigma = cov 
        
        rout( (x - mean)^T * inv_cov * (x - mean)  )
        input must be 1d vector
        """
        try:
            inv_cov = np.linalg.inv(cov)
        except:
            inv_cov = np.linalg.pinv(cov)

        input_vec = input_vec - mean

        # cov^-1 * (x-man)
        l = np.dot(input_vec, inv_cov)

        # 2-dim gaussian mahalanobis distance 
        # rout( (x - mean)^T * inv_cov * (x - mean)  )
        dist = np.sqrt(np.dot(l, input_vec.T).diagonal())
        return dist

    @staticmethod
    def new_coordinates(data, eigenvectors):
        """
        projection to eigenspace
        """
        for i in range(eigenvectors.shape[0]):
            if i == 0:
                new = [
                    data.dot(eigenvectors.T[i])
                    ]
            else:
                new = np.concatenate((new, [data.dot(eigenvectors.T[i])]), axis=0)
        return new.T        
          

    def visualize_eigenface(self, k_list):
        from sklearn.decomposition import PCA
        data_dict = self.data.test

        self.pca_data_dict = dict()
        prefix = 'eigen_{:03d}.jpg'

        for i, key in enumerate(self.data.train.keys()):
            if i ==0:
                train_cat = self.data.train[key]
            else:
                train_cat = np.concatenate((train_cat, self.data.train[key]))

        for k in k_list:
            print(" K ", k)
            save_root = "eigenfaces"
            if not os.path.isdir(save_root):
                os.mkdir(save_root)
            test = self.data.test[1][0] 
            test = np.expand_dims(test, axis=0)
            pca = PCA(
                n_components=k, 
                whiten=True
            )
            pca.fit(train_cat)
            test_transformed = pca.transform(test)
            reconstructed = pca.inverse_transform(test_transformed)
            reconstructed = np.reshape(reconstructed, (self.data.w, self.data.h, 1))
            cv2.imshow("{}".format(k), reconstructed/255)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(save_root, "eigen_k[{}].png".format(k)), reconstructed)


    def visualize_faces2d(self):
        from sklearn.decomposition import PCA
        data_dict = self.data.train
        colors = cm.rainbow(np.linspace(0, 1, len(self.data.subject_ids)))
        self.pca_data_dict = dict()

        for i, key in enumerate(self.data.train.keys()):
            if i ==0:
                train_cat = self.data.train[key]
            else:
                train_cat = np.concatenate((train_cat, self.data.train[key]))

        pca = PCA(
            n_components=2, 
            whiten=True
        )
        pca.fit(train_cat)
        proj = pca.transform(train_cat)
        num_train_per_id = 9
        for i, subject in enumerate(self.data.subject_ids):
            for j in range(num_train_per_id):
                # 1. proj 2-dim space
                proj_face_vec = proj[i*num_train_per_id+j]
                plt.scatter(
                    proj_face_vec[0],
                    proj_face_vec[1], 
                    c = colors[i],
                    s = 10
                )
        plt.show()

    def faces_PCA_train(self):
        from sklearn.decomposition import PCA
        data_dict = self.data.train
        self.pca_data_dict = dict()

        for i, key in enumerate(self.data.train.keys()):
            if i ==0:
                train_cat = self.data.train[key]
            else:
                train_cat = np.concatenate((train_cat, self.data.train[key]))
        print("- Train data : ", train_cat.shape) # (360, 2576)

        pca = PCA(
            n_components=self.n_component, 
            whiten=True
        )
        pca.fit(train_cat)
        self.component = pca.components_
        self.mean = pca.mean_
        self.meanface = pca.mean_.reshape(self.data.img_list[0].shape)
    


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