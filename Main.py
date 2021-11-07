from genericpath import exists
import os, sys
import tensorflow as tf
import numpy as np
import time
from Reader import Reader
from Model import Model
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

BATCH_SIZE = 32
LOG_FREQ = 1000
TEST_FREQ = LOG_FREQ * 2
SAVE_FREQ = LOG_FREQ * 5
LOG_DIR = os.path.join(os.curdir, "log")
MOD = 3

class Main:
    def __init__(self):
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

        self.reader = Reader()
        
        max_itr = int(400 * self.reader.train_size / BATCH_SIZE) + 1

        model_index = 13
        while True:
            self.model = Model(model_index)

            log_dir = os.path.join(LOG_DIR, "model_" + str(model_index))

            if self.model.mod_support > 0:
                log_dir += "_mod_" + str(MOD)

            if os.path.exists(log_dir):
                print("Output dir already exist")
                return
            else:
                os.mkdir(log_dir)
            
            print(self.model.get_model_description())

            with open(os.path.join(log_dir, "log.txt"), "a") as fp:
                fp.write(self.model.get_model_description() + "\n")
            
            self.train(max_itr, log_dir)

            model_index += 1

            if len(self.model.models) == model_index:
                break

    def train(self, max_itr, log_dir):
        itr = 0
        
        read_time = 0
        train_time = 0

        start_time = time.time()

        while itr < max_itr:
            read_start = time.time()
            x, y = self.reader.get_train_batch(BATCH_SIZE)
            read_time += time.time() - read_start

            train_start = time.time()
            if self.model.mod_support == 0:
                loss = self.model.train(x)
            elif self.model.mod_support == 1:
                loss = self.model.train_with_mod(x, (y % MOD).astype("float"))
            elif self.model.mod_support == 2:
                loss = self.model.train_with_mod_2(x, (y % MOD).astype("float"))
            train_time += time.time() - train_start

            if itr % LOG_FREQ == 0:
                if itr % TEST_FREQ == 0:
                    test_loss, test_images = self.test()

                epoch = itr * BATCH_SIZE / self.reader.train_size
                ellapse_hour = (time.time() - start_time) / 60 / 60

                log_str = f"{itr:7d}, {epoch:.2f} epoch, train_loss {loss:.5f}, test_loss {test_loss:.5f}, train {train_time:.3f}s, read {read_time:.3f}s, ellapse {ellapse_hour:.2f}hour"
                print(log_str)
                read_time = 0
                train_time = 0

                if itr % SAVE_FREQ == 0:
                    with open(os.path.join(log_dir, "log.txt"), "a") as fp:
                        fp.write(log_str + "\n")
                    
                    for i in range(3):
                        test_images[i].save(os.path.join(log_dir, str(itr) + "_" + str(i) + ".jpg"))

                    self.model.save_model(os.path.join(log_dir, str(itr)))

            itr += 1

    def test(self):
        x_test, y_test = self.reader.get_test_batch(int(BATCH_SIZE / 2))

        if self.model.mod_support == 0:
            enc, pred = self.model.predict(x_test)
            loss = self.model.get_loss(x_test, pred)
        elif self.model.mod_support == 1:
            mod = (y_test % MOD).astype("float")
            enc, pred, pred_mod = self.model.predict_with_mod(x_test, mod)
            loss = self.model.get_loss_with_mod(x_test, mod, pred, pred_mod)
        elif self.model.mod_support == 2:
            mod = (y_test % MOD).astype("float")
            enc, pred, pred_mod = self.model.predict_with_mod_2(x_test, mod)
            loss = self.model.get_loss_with_mod(x_test, mod, pred, pred_mod)

        combined = np.concatenate((x_test, pred), axis=2)
        images = get_image(combined)

        return loss, images
    
def get_image(data):
    data = np.array(data)

    # Remove last dimention if one
    if data.shape[-1] == 1:
        data = data.reshape(data.shape[:-1])

    # Add batch dimention if not available
    if len(data.shape) == 2:
        data = data.reshape((1,) + data.shape)
    
    images = []
    for i in range(data.shape[0]):
        d = data[i]
        d = (d * 255).astype("int")
        img = Image.fromarray(d)
        img = img.convert("RGB")
        images.append(img)
    
    return images

class FeatureExtracter:
    def __init__(self, model_index, checkpoint_index):
        from mpl_toolkits.mplot3d import Axes3D

        self.reader = Reader()
        
        self.model = Model(model_index)

        log_dir = os.path.join(LOG_DIR, "model_" + str(model_index))
        
        if self.model.mod_support > 0:
            log_dir += "_mod_" + str(MOD)

        checkpoint_path = os.path.join(log_dir, str(checkpoint_index))
        self.model.load_model(checkpoint_path)

        if self.model.mod_support == 0:
            enc, _ = self.model.predict(self.reader.x_test)
        elif self.model.mod_support == 1:
            enc, pred, pred_mod = self.model.predict_with_mod(self.reader.x_test, (self.reader.y_test % MOD).astype("float"))
        elif self.model.mod_support == 2:
            enc, pred, pred_mod = self.model.predict_with_mod_2(self.reader.x_test, (self.reader.y_test % MOD).astype("float"))
        
        enc = np.array(enc)

        fig = plt.figure()
        ax = Axes3D(fig)

        stride = 1
        for c in range(10):
            marker = "+"
            if c == 4: marker = "1"
            if c == 5: marker = "2"
            if c == 7: marker = "."
            if c == 9: marker = "x"
            x = enc[self.reader.y_test==c][::stride,0]
            y = enc[self.reader.y_test==c][::stride,1]
            z = enc[self.reader.y_test==c][::stride,2]
            #plt.scatter(x, y, label=str(c), marker="+") # 2D scatter
            ax.scatter(x, y, z, label=str(c), marker=marker) # 3D scatter
        
        plt.ion()
        plt.legend()
        plt.show()
        
        num_cluster = 12
        cluster = KMeans(num_cluster, max_iter=3000).fit_predict(enc)

        y_onehot = np.eye(10)[self.reader.y_test]
        y_onehot_accum = np.zeros([num_cluster, 10]).astype("int")

        # Make y_onehot_accum that shape of [cluster, y]
        for i, yy in enumerate(y_onehot):
            c = cluster[i]
            y_onehot_accum[c] = y_onehot_accum[c] + y_onehot[i]

        self.save_vector(log_dir, enc, self.reader.y_test, cluster, y_onehot_accum)

        # Save image for cluster dir
        print("Save image")
        for c in range(num_cluster):
            cluster_dir = os.path.join(log_dir, "cluster_" + str(c))
            if not os.path.exists(cluster_dir):
                os.mkdir(cluster_dir)
        
        for i, c in enumerate(cluster):
            y = self.reader.y_test[i]
            file_name = os.path.join(log_dir, "cluster_" + str(c), str(y) + "_" + str(i) + ".jpg")
            image = get_image(self.reader.x_test[i])
            image[0].save(file_name)

    def save_vector(self, log_dir, enc, y, cluster, y_onehot_accum):
        vector_file = os.path.join(log_dir, "encoder_out.txt")

        with open(vector_file, "a") as fp:
            for batch in range(enc.shape[0]):
                fp.write(str(y[batch]) + "," + str(cluster[batch]) + ",")
                for i in range(enc.shape[1]):
                    fp.write(str(enc[batch][i]) + ",")
                fp.write("\n")
            
            fp.write("\n")
            for i in range(y_onehot_accum.shape[0]):
                fp.write(f"Cluster{i:02}," + str(y_onehot_accum[i]) + "\n")
            

if __name__ == "__main__":
    #Main()
    FeatureExtracter(13, 750000)