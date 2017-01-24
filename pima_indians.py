# Jack Schefer, began 1.23.17
# taken from http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#
#
from __future__   import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from numpy        import random, loadtxt
#
#
def main():
    #
    # 1.
    random.seed( 7 )
    dataset = loadtxt( 'data/pima-indians-diabetes.csv', delimiter = ',' )
    x = dataset[:,0:8]
    y = dataset[:,8]
    #
    #
    try:
        #
        with open('saved_models/pima_indians.json', 'r') as f:
            model = model_from_json( f.read() )
        #
        model.load_weights('saved_models/pima_indians.h5')
        print('loaded from disk')
        #
    except IOError:
        # 2.
        model = Sequential()
        model.add( Dense( 12, input_dim = 8, init = 'uniform', activation = 'relu' ) )
        model.add( Dense( 8, init = 'uniform', activation = 'relu' ) )
        model.add( Dense( 1, init = 'uniform', activation = 'sigmoid' ))
        #
        # 3.
        model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = [ 'accuracy' ] )
        #
        # 4.
        model.fit( x, y, nb_epoch = 150, batch_size = 10 )
        #
        model_json = model.to_json()
        with open('saved_models/pima_indians.json', 'w' ) as f:
            f.write( model_json )
        #
        model.save_weights('saved_models/pima_indians.h5')
        print('saved to disk')
    # 5.
    #scores = model.evaluate( x, y )
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #
    preds = model.predict( x )
    rounded = [ round(s) for s in preds ]
    wrong = 0
    for i in range(len(rounded)): 
        if rounded[i] != y[i]: wrong += 1
    #
    print('wrong:', wrong, '\ttotal:', len(rounded))
    #
#
#
if __name__ == '__main__':
    main()
#
# End of file.
