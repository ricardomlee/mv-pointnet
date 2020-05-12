from data_loader import DataGenerator
from model_cls import MVPointNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from schedules import onetenth_50_75
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import globals as _g

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def main():
    nb_classes = _g.NUM_CLASSES

    train_file = './data/ModelNetS/mv_train.h5'
    test_file = './data/ModelNetS/mv_test.h5'

    epochs = 100
    batch_size = 32

    train = DataGenerator(train_file, batch_size, nb_classes, train=True)
    val = DataGenerator(test_file, batch_size, nb_classes, train=False)

    model = MVPointNet(nb_classes)
    model.summary()
    lr = 0.0001
    adam = Adam(lr=lr)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if not os.path.exists('./resultsmv/'):
        os.mkdir('./resultsmv/')
    checkpoint = ModelCheckpoint('./resultsmv/mvpointnet.h5', monitor='val_acc',
                                 save_weights_only=True, save_best_only=True,
                                 verbose=1)
    history = model.fit_generator(train.generator(),
                                  steps_per_epoch=1600 // batch_size,
                                  epochs=epochs,
                                  validation_data=val.generator(),
                                  validation_steps=320 // batch_size,
                                  callbacks=[checkpoint, onetenth_50_75(lr)],
                                  verbose=1)

    plot_history(history, './resultsmv/')
    save_history(history, './resultsmv/')
    model.save_weights('./resultsmv/mvpointnet_weights.h5')


if __name__ == '__main__':
    main()
