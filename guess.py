from idw import IDW
from keras.models import model_from_json
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import random
import time

directory = "Earth"
im_size = 128
suffix = 'jpg'
model_num = 15

def loadModel(name, num):

    file = open("Models/"+name+".json", 'r')
    json = file.read()
    file.close()

    mod = model_from_json(json)
    mod.load_weights("Models/"+name+"_"+str(num)+".h5")

    return mod

class dataGenerator(object):

    def __init__(self, loc, flip = True, suffix = 'png'):
        self.flip = flip
        self.suffix = suffix
        self.files = []
        self.n = 1e10

        print("Importing Images...")

        try:
            os.mkdir("data/" + loc + "-npy-" + str(im_size))
        except:
            self.load_from_npy(loc)
            return

        for dirpath, dirnames, filenames in os.walk("data/" + loc):
            for filename in [f for f in filenames if f.endswith("."+str(self.suffix))]:
                print('\r' + str(len(self.files)), end = '\r')
                fname = os.path.join(dirpath, filename)
                temp = Image.open(fname).convert(cmode)
                if not size_adjusted:
                    temp = temp.resize((im_size, im_size), Image.BILINEAR)
                temp = np.array(temp, dtype='uint8')
                self.files.append(temp)
                if self.flip:
                    self.files.append(np.flip(temp, 1))

        self.files = np.array(self.files)
        np.save("data/" + loc + "-npy-" + str(im_size) + "/data.npy", self.files)

        self.n = self.files.shape[0]

        print("Found " + str(self.n) + " images in " + loc + ".")

    def load_from_npy(self, loc):

        print("Loading from .npy files.")

        self.files = np.load("data/" + str(loc) + "-npy-" + str(im_size) + "/data.npy")

        self.n = self.files.shape[0]


    def get_batch(self, num):

        idx = np.random.randint(0, self.n, num)
        out = []

        for i in range(num):
            out.append(self.files[idx[i]])

        return np.array(out).astype('float32') / 255.0

data = dataGenerator(directory, suffix = suffix)
encoder = loadModel("enc", model_num)
generator = loadModel("gen", model_num)


def game(num = 10):
    total_rounds = 0
    total_score = 0

    tpoints = []
    tvalues = []
    tdiff = []
    diffma = []

    dim_weights = np.ones([64])

    while True:
        images = data.get_batch(num)
        points = encoder.predict(images)

        for i in range(images.shape[0]):
            if total_rounds > 2:
                prediction = round(IDW(points[i], tpoints, tvalues, exp = 35, dim_weights = dim_weights), 1)
            else:
                prediction = 5

            print("Rate this image (from 1 to 10):")
            plt.imshow(images[i])
            plt.show(block = False)

            real = float(input())

            diff = round(abs(prediction - real), 1)

            print("You rated this image a " + str(real) + ", I guessed you would rate it " + str(prediction) + ".")

            if diff <= 1:
                print("That's a difference of only " + str(diff) + "! Score!")
            else:
                print("That's a difference of " + str(diff) + "... No score.")



            print()

            total_rounds = total_rounds + 1
            tpoints.append(points[i])
            tvalues.append(real)
            tdiff.append(diff)
            diffma.append(np.mean(tdiff[-20:]))

        dim_weights = evolve(tpoints, tvalues, dim_weights)

        print("Average difference (all rounds): " + str(np.mean(tdiff)))
        print("Average difference (last 20): " + str(np.mean(tdiff[-20:])))

        plt.close()
        plt.figure(2)
        plt.plot(diffma)
        plt.show(block = False)
        plt.figure(1)

        print("\nDimension weights:")
        print(np.round(dim_weights * 100) / 100)
        print()

        generate(tpoints, tvalues, dim_weights)

        print()
        print()

def evolve(x, y, params, rounds = 5, population = 50, heat = 0.1):

    tparams = params.copy()

    print("Evolving parameters.")

    train_x = x[::2]
    train_y = y[::2]

    test_x = x[1::2]
    test_y = y[1::2]



    for r in range(rounds):

        fitness = []
        new_params = []

        for i in range(population):
            #Mutate
            new_params.append(tparams + np.random.normal(0.0, heat, params.shape))
            fitness.append(0)

            #Make Predictions Again
            for j in range(len(test_x)):
                pred = IDW(test_x[j], train_x, train_y, exp = 35, dim_weights = new_params[-1])
                fitness[-1] = fitness[-1] + (pred - test_y[j]) ** 2

        best_one = np.argmin(fitness)

        print("Best fitness: " + str(fitness[best_one] / len(train_x)))

        tparams = new_params[best_one]

    return tparams

def generate(x, y, params, rounds = 5, population = 50):

    print("Generating Image...")

    tlatent = np.random.normal(0.0, 1.0, [64])

    for r in range(rounds):

        fitness = []
        new_latent = []

        for i in range(population):
            #Mutate
            if r > 0:
                new_latent.append(tlatent + np.random.normal(0.0, 0.01, tlatent.shape))
            else:
                new_latent.append(np.random.normal(0.0, 1.0, tlatent.shape))
            score = IDW(new_latent[-1], x, y, exp = 35, dim_weights = params)
            fitness.append(score)

        best_one = np.argmax(fitness)

        print("Best fitness: " + str(fitness[best_one]))

        tlatent = new_latent[best_one]

    image = generator.predict(np.array([tlatent]))[0]

    plt.figure(3)
    plt.imshow(image)
    plt.show(block=False)
    plt.figure(1)

def similarity(points):
    image = random.randint(0, data.files.shape[0])

    similarity = []

    vline = np.ones([128, 10, 3]) * 255

    for i in range(0, points.shape[0]):
        similarity.append(np.linalg.norm(points[image] - points[i]))

    best = np.argsort(similarity)[1:10]

    big_image = np.concatenate([data.files[image], vline, data.files[best[0]], data.files[best[1]], data.files[best[2]],
                                data.files[best[3]], data.files[best[4]], data.files[best[5]], data.files[best[6]]], axis = 1)

    return big_image

def cluster(points, means = 8):
    kk = KMeans(n_clusters = means)
    kk.fit(points)

    labels = kk.predict(points)

    r = []

    for i in range(means):
        row = []
        while(len(row) < 8):
            image = random.randint(0, data.files.shape[0] - 1)
            if labels[image] == i:
                row.append(data.files[image])

        r.append(np.concatenate(row, axis=1))

    c = np.concatenate(r, axis=0)

    x = Image.fromarray(c)
    x.save('Results/clusters.png')

def createDiagram(points):
    r = []
    for i in range(8):
        r.append(similarity(points).astype('uint8'))

    c = np.concatenate(r, axis=0)

    x = Image.fromarray(c)
    x.save('Results/diagram.png')

#points = encoder.predict(data.files.astype('float32') / 255.0, batch_size = 64, verbose = 1)
#np.save("points.npy", points)

points = np.load("points.npy")

createDiagram(points)
