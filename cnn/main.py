# EXTERNAL IMPORT(S)
import time, datetime, ntpath
import matplotlib.pyplot as plt, matplotlib.pyplot as ply
import torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# INTERNAL IMPORT(S)
from .datasets import *

# GLOBAL STATIC VARS
TRAIN_DIR = '/train'
VALIDATION_DIR = '/validation'

PATCH_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 1

LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
LOG_INTERVAL = 50
EPOCHS = 30

class Base:
    def __init__(self, args, network, weightsDir):
        self.args = args
        self.weights = weightsDir
        self.network = network.cuda()
        self.load()

    def load(self):
        self.network.load_state_dict(torch.load(self.weights))

    def save(self):
        print('Saved Directory: "{}"'.format(self.weights))
        torch.save(self.network.state_dict(), self.weights)


class PatchLevel(Base):
    def __init__(self, args, network):
        super(PatchLevel, self).__init__(args, network, args.pretrained_path + '/weights_' + network.name() + '.pth')

    def train(self):
        self.network.train()
        print('Patch Level TRAIN Start: {}\n'.format(time.strftime('%m/%d %H:%M:%S')))

        trainLoader = DataLoader(
            dataset=PatchLevelData(path=self.args.data_dir + TRAIN_DIR, stride=self.args.patch_stride, rotate=True, flip=True, enhance=True),
            batch_size=PATCH_BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        best = self.validate(verbose=False)
        mean = 0
        epoch = 0

        for epoch in range(1, EPOCHS + 1):
            self.network.train()
            scheduler.step()
            startTime = datetime.datetime.now()

            correct = 0
            total = 0

            for index, (images, labels) in enumerate(trainLoader):

                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                if index > 0 and index % LOG_INTERVAL == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        EPOCHS,
                        index * len(images),
                        len(trainLoader.dataset),
                        (100. * index / len(trainLoader)),
                        loss.data[0],
                        (100 * correct / total)
                    ))

            print('\nepoch {}, time: {}'.format(epoch, datetime.datetime.now() - startTime))
            acc = self.validate()
            mean += acc
            if acc > best:
                best = acc
            self.save()

        print('\nbest accuracy: {}, mean accuracy: {}\n'.format(best, mean // epoch))

    def validate(self, verbose=True):
        self.network.eval()

        testLoss = 0
        correct = 0
        classes = len(CATEGORY)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        test_loader = DataLoader(
            dataset=PatchLevelData(path=self.args.data_dir + VALIDATION_DIR, stride=self.args.patch_stride),
            batch_size=PATCH_BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        print('\n--- Evaluating ---')

        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            output = self.network(Variable(images, volatile=True))

            testLoss += F.nll_loss(output, Variable(labels), size_average=False).data[0]
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)

            for label in range(classes):
                valLabels = labels == label
                predLabels = predicted == label
                tp[label] += torch.sum(valLabels == (predLabels * 2 - 1))
                tpfp[label] += torch.sum(predLabels)
                tpfn[label] += torch.sum(valLabels)

        for label in range(classes):
            precision[label] += (tp[label].cpu().numpy() / (tpfp[label].cpu().numpy() + 1e-8))
            recall[label] += (tp[label].cpu().numpy() / (tpfn[label].cpu().numpy() + 1e-8))
            f1[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label] + 1e-8)

        testLoss /= len(testLoader.dataset)
        acc = 100. * correct / len(testLoader.dataset)

        print('mean loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(testLoss, correct, len(testLoader.dataset), (100. * correct / len(PatchLevelData.dataset))))
        for label in range(classes):
            print('{}:  \t precision: {:.2f},  recall: {:.2f},  f1: {:.2f}\n'.format(CATEGORY[label], precision[label], recall[label], f1[label]))
        return acc

    def test(self, path, verbose=True):
        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE, augment=False)
        dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        startTime = datetime.datetime.now()
        res = []

        for index, (image, fileName) in enumerate(dataLoader):
            image = image.squeeze()
            image = image.cuda()

            output = self.network(Variable(image))
            _, predicted = torch.max(output.data, 1)
            majorityVote = 3 - np.argmax(np.sum(np.eye(4)[np.array(predicted.cpu().numpy()).reshape(-1)], axis=0)[::-1])

            res.append([majorityVote, fileName[0]])

            np.sum(output.data.cpu().numpy(), axis=0)
            print('{}) \t {} \t {}'.format(str(index + 1).rjust(2, '0'),
                                            CATEGORY[majorityVote].ljust(8),
                                            ntpath.basename(fileName[0])))

        print('\nProcess/Inference time: {}\n'.format(datetime.datetime.now() - startTime))

        return res

    def output(self, tempTensor):
        self.network.eval()
        res = self.network.features(Variable(tempTensor, volatile=True))
        return res.squeeze()

    def visualize(self, path, channel=0):
        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE)
        dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        for index, (image, fileName) in enumerate(dataLoader):
            image = image[0].cuda()
            patches = self.output(image)
            output = patches.cpu().data.numpy()
            map = np.zeros((3 * 64, 4 * 64))

            for i in range(16):
                row = i // 4
                col = i % 4
                map[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = output[i]

            if len(map.shape) > 2:
                map = map[channel]

            with Image.open(fileName[0]) as roc_graph:
                ply.subplot(121)
                ply.axis('off')
                ply.imshow(np.array(roc_graph))
                ply.subplot(122)
                ply.imshow(map, cmap='gray')
                ply.axis('off')

                ply.show()


class ImageLevel(Base):
    def __init__(self, args, patchLevelNetwork, imageLevelNetwork):
        super(ImageLevel, self).__init__(args, imageLevelNetwork, args.pretrained_path + '/weights_' + imageLevelNetwork.name() + '.pth')

        self.patchLevel = PatchLevel(args, patchLevelNetwork)
        self.testLoader = None

    def train(self):
        self.network.train()
        trainLoader = self.patch_loader(self.args.data_dir + TRAIN_DIR, True)

        print('Image Level TRAIN Start: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        best = self.validate(verbose=False)
        mean = 0
        epoch = 0

        for epoch in range(1, EPOCHS + 1):
            self.network.train()
            startTime = datetime.datetime.now()
            correct = 0
            total = 0

            for index, (images, labels) in enumerate(trainLoader):
                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                if index > 0 and index % 10 == 0:
                    print('epoch: {}/{} [{}/{} ({:.0f}%)]\tloss: {:.6f}, accuracy: {:.2f}%'.format(
                        epoch,
                        EPOCHS,
                        index * len(images),
                        len(trainLoader.dataset),
                        100. * index / len(trainLoader),
                        loss.data[0],
                        100 * correct / total
                    ))

            print('\nepoch {}, time: {}'.format(epoch, datetime.datetime.now() - startTime))
            acc = self.validate()
            mean += acc
            if acc > best:
                best = acc
                self.save()

        print('\nbest accuracy: {}, mean accuracy: {}\n'.format(best, mean // epoch))

    def validate(self, verbose=True, roc=False):
        self.network.eval()

        if self.testLoader is None:
            self.testLoader = self.patch_loader(self.args.data_dir + VALIDATION_DIR, False)

        valLoss = 0
        correct = 0
        classes = len(CATEGORY)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        print('\n--- Evaluating ---')

        labelsTrue = []
        labelsPred = np.empty((0, 4))

        for images, labels in self.testLoader:
            images = images.cuda()
            labels = labels.cuda()
            output = self.network(Variable(images, volatile=True))

            valLoss += F.nll_loss(output, Variable(labels), size_average=False).data[0]
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)

            labelsTrue = np.append(labelsTrue, labels)
            labelsPred = np.append(labelsPred, torch.exp(output.data).cpu().numpy(), axis=0)

            for label in range(classes):
                valLabels = labels == label
                predLabels = predicted == label
                tp[label] += torch.sum(valLabels == (predLabels * 2 - 1))
                tpfp[label] += torch.sum(predLabels)
                tpfn[label] += torch.sum(valLabels)

        for label in range(classes):
            precision[label] += (tp[label].cpu().numpy() / (tpfp[label].cpu().numpy() + 1e-8))
            recall[label] += (tp[label].cpu().numpy() / (tpfn[label].cpu().numpy() + 1e-8))
            f1[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label] + 1e-8)

        valLoss /= len(self.testLoader.dataset)
        acc = 100. * correct / len(self.testLoader.dataset)

        if roc == 1:
            labelsTrue = label_binarize(labelsTrue, classes=range(classes))
            for lbl in range(classes):
                fpr, tpr, _ = roc_curve(labelsTrue[:, lbl], labelsPred[:, lbl])
                rocAuc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label='{} (AUC: {:.1f})'.format(CATEGORY[lbl], rocAuc * 100))

            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('FP (False Positive) Rate')
            plt.ylabel('TP (True Positive) Rate')
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
            plt.title('ROC (Receiver Operating Characteristic)')
            plt.show()

        print('average loss: {:.3f}, accuracy: {}/{} ({:.0f}%)'.format(valLoss, correct, len(self.testLoader.dataset), acc))

        for label in range(classes):
            print('{}:  \t precision: {:.2f},  recall: {:.2f},  F1: {:.2f}\n'.format(CATEGORY[label], precision[label], recall[label], f1[label]))

        return acc

    def test(self, path, verbose=True):
        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE)
        dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        startTime = datetime.datetime.now()

        res = []

        for index, (image, fileName) in enumerate(dataLoader):
            nBins, nPatches = image.shape[1], image.shape[2]
            image = image.view(-1, 3, PATCH_SIZE, PATCH_SIZE)
            image = image.cuda()

            patches = self.patchLevel.output(image)
            patches = patches.view(nBins, -1, 64, 64)
            patches = patches.cuda()

            output = self.network(patches)
            _, predicted = torch.max(output.data, 1)

            majorityVote = 3 - np.argmax(np.sum(np.eye(4)[np.array(predicted.cpu().numpy()).reshape(-1)], axis=0)[::-1])

            confidence = np.sum(np.array(predicted) == majorityVote) / nBins if True else torch.max(torch.exp(output.data))
            confidence = np.round(confidence * 100, 2)

            res.append([majorityVote, confidence, fileName[0]])

            print('{}) {} ({}%) \t {}'.format(str(index).rjust(2, '0'), CATEGORY[majorityVote], confidence, ntpath.basename(fileName[0])))
            print('\nProcess/Inference time: {}\n'.format(datetime.datetime.now() - startTime))

        return res

    def patch_loader(self, path, augment):
        imagesPath = '{}/{}_images.npy'.format(self.args.pretrained_path, self.network.name())
        labelsPath = '{}/{}_labels.npy'.format(self.args.pretrained_path, self.network.name())

        if self.args.debug and augment and os.path.exists(imagesPath):
            npImages = np.load(imagesPath)
            npLabels = np.load(labelsPath)

        else:
            dataset = ImageLevelData(
                path=path,
                stride=PATCH_SIZE,
                flip=augment,
                rotate=augment,
                enhance=augment)

            outputLoader = DataLoader(dataset=dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=True, num_workers=4)
            outputImages = []
            outputLabels = []

            for index, (images, labels) in enumerate(outputLoader):
                if index > 0 and index % 10 == 0:
                    print('{} image successfully loaded'.format(int((index * IMAGE_BATCH_SIZE) / dataset.augment_size)))

                images = images.cuda()
                size = images.shape[0]
                res = self.patchLevel.output(images.view((-1, 3, 512, 512)))
                res = res.view((IMAGE_BATCH_SIZE, -1, 64, 64)).data.cpu().numpy()

                for i in range(size):
                    outputImages.append(res[i])
                    outputLabels.append(labels.numpy()[i])

            npImages = np.array(outputImages)
            npLabels = np.array(outputLabels)

            np.save(imagesPath, npImages)
            np.save(labelsPath, npLabels)

        images = torch.from_numpy(npImages)
        labels = torch.from_numpy(npLabels).squeeze()

        return DataLoader(dataset=TensorDataset(images, labels), batch_size=PATCH_BATCH_SIZE, shuffle=True, num_workers=2)
