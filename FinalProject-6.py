import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("Tensorflow Version: %s"%tf.__version__)

#Load data
#npzfile = np.load("F:\\Education\\RPI\\2017-Spring\\DeepLearning\\FinalProject\\train_and_val.npz")
#npzfile = np.load("/media/yogish/Local Disk 2/Education/RPI/2017-Spring/DeepLearning/FinalProject/train_and_val.npz")
npzfile = np.load("./train_and_val.npz")

train_eye_left_orig = npzfile["train_eye_left"]
train_eye_right_orig = npzfile["train_eye_right"]
train_face_orig = npzfile["train_face"]
train_face_mask_orig = npzfile["train_face_mask"]
train_y_orig = npzfile["train_y"]
val_eye_left_orig = npzfile["val_eye_left"]
val_eye_right_orig = npzfile["val_eye_right"]
val_face_orig = npzfile["val_face"]
val_face_mask_orig = npzfile["val_face_mask"]
val_y_orig = npzfile["val_y"]

#Parameters-input
numSamplesTrain = train_face_orig.shape[0]
numSamplesTest = val_face_orig.shape[0]
imageWidth = train_face_orig.shape[1]
imageHeight = train_face_orig.shape[2]
imageChannels = train_face_orig.shape[3]
maskWidth = train_face_mask_orig.shape[1]
maskHeight = train_face_mask_orig.shape[2]

print("numSamplesTrain: %d\tnumSamplesTest: %d"%(numSamplesTrain,numSamplesTest))
print("imageSize: [%d,%d,%d]"%(imageHeight,imageWidth,imageChannels))
print("maskSize: [%d,%d]"%(maskHeight,maskWidth))

#Parameters-filters
#Eye
conv_E1_filterSize = 5#Output:60x60x32; 30x30x32 pool
conv_E1_numFilters = 32
conv_E2_filterSize = 5#Output:26x26x32; 13x13x32 pool
conv_E2_numFilters = 32
conv_E3_filterSize = 3#Output:11x11x64; No pool
conv_E3_numFilters = 64

#Face
conv_F1_filterSize = 5
conv_F1_numFilters = 32
conv_F2_filterSize = 5
conv_F2_numFilters = 32
conv_F3_filterSize = 3
conv_F3_numFilters = 64

fc_E1_numFilters = 256
fc_E2_numFilters = 256
fc_F1_numFilters = 256
fc_F2_numFilters = 512
fc_FM1_numFilters = 256
fc_FM2_numFilters = 128

fc_EFM1_numFilters = 256
fc_EFM2_numFilters = 2

#Learning rate
lr = 0.001

#Placeholders
eye_left = tf.placeholder(tf.float32,[None,imageHeight,imageWidth,imageChannels])#Input;left eye
eye_right = tf.placeholder(tf.float32,[None,imageHeight,imageWidth,imageChannels])#Input;right eye
face = tf.placeholder(tf.float32,[None,imageHeight,imageWidth,imageChannels])#Input;face
face_mask = tf.placeholder(tf.float32,[None,maskHeight,maskWidth])#Input;face mask
y = tf.placeholder(tf.float32,[None,2])#Output;gaze position

#Common function definitions
#Perform convolution
def fn_conv(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    return (conv + biases)

#Perform convolution and apply relu activation
def fn_conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.relu(conv + biases)

#Perform convolution, apply relu activation and apply max pooling
def fn_conv_relu_pool(input, kernel_shape, bias_shape, pool_shape, pool_stride):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(conv + biases)
    return tf.cast(tf.nn.max_pool(relu,pool_shape,pool_stride,"VALID"),tf.float32)

#Fully connected layer
def fn_fullyconn(input, numNodesInput, numNodesOutput):
    # Create variable named "weights".
    weights = tf.get_variable("weights", [numNodesInput,numNodesOutput], initializer=tf.random_uniform_initializer(0.0,0.001))
    # Create variable named "biases".
    biases = tf.get_variable("biases", numNodesOutput, initializer=tf.constant_initializer(0.0))
    return tf.add(tf.matmul(input,weights),biases)

#Main function containing architecture
def my_image_filter(left_eye,right_eye,face,face_mask):
#Initial convolutional layers for eyes
    with tf.variable_scope("conv_E1_left"):
        # Variables created here will be named "conv_E1/weights", "conv_E1/biases".
        out_left_convE1 = fn_conv_relu_pool(left_eye,
                                  [conv_E1_filterSize,conv_E1_filterSize,imageChannels,conv_E1_numFilters],
                                  [conv_E1_numFilters],[1,2,2,1],[1,2,2,1])
        print("out_left_convE1: %s"%(out_left_convE1))
    with tf.variable_scope("conv_E1_right"):
        out_right_convE1 = fn_conv_relu_pool(right_eye,
                                  [conv_E1_filterSize,conv_E1_filterSize,imageChannels,conv_E1_numFilters],
                                  [conv_E1_numFilters],[1,2,2,1],[1,2,2,1])
        print("out_right_convE1: %s"%(out_right_convE1))
    with tf.variable_scope("conv_E2_left"):
        # Variables created here will be named "conv_E2/weights", "conv_E2/biases".
        out_left_convE2 = fn_conv_relu_pool(out_left_convE1,
                                  [conv_E2_filterSize,conv_E2_filterSize,conv_E1_numFilters,conv_E2_numFilters],
                                  [conv_E2_numFilters],[1,2,2,1],[1,2,2,1])
        print("out_left_convE2: %s"%(out_left_convE2))
    with tf.variable_scope("conv_E2_right"):
        out_right_convE2 = fn_conv_relu_pool(out_right_convE1,
                                  [conv_E2_filterSize,conv_E2_filterSize,conv_E1_numFilters,conv_E2_numFilters],
                                  [conv_E2_numFilters],[1,2,2,1],[1,2,2,1])
        print("out_right_convE2: %s"%(out_right_convE2))
    with tf.variable_scope("conv_E3_left"):
        # Variables created here will be named "conv_E3/weights", "conv_E3/biases".
        out_left_convE3 = fn_conv_relu(out_left_convE2,
                                  [conv_E3_filterSize,conv_E3_filterSize,conv_E2_numFilters,conv_E3_numFilters],
                                  [conv_E3_numFilters])
        print("out_left_convE3: %s"%(out_left_convE3))
    with tf.variable_scope("conv_E3_right"):
        out_right_convE3 = fn_conv_relu(out_right_convE2,
                                  [conv_E3_filterSize,conv_E3_filterSize,conv_E2_numFilters,conv_E3_numFilters],
                                  [conv_E3_numFilters])
        print("out_right_convE3: %s"%(out_right_convE3))
#Initial convolutional layers for face
    with tf.variable_scope("conv_F1"):
        # Variables created here will be named "conv_E1/weights", "conv_E1/biases".
        out_face_convF1 = fn_conv_relu_pool(face,
                                            [conv_F1_filterSize,conv_F1_filterSize,imageChannels,conv_F1_numFilters],
                                            [conv_F1_numFilters],[1,2,2,1],[1,2,2,1])
        print("out_face_convF1: %s"%(out_face_convF1))
    with tf.variable_scope("conv_F2"):
        # Variables created here will be named "conv_E1/weights", "conv_E1/biases".
        out_face_convF2 = fn_conv_relu_pool(out_face_convF1,
                                            [conv_F2_filterSize,conv_F2_filterSize,conv_F1_numFilters,conv_F2_numFilters],
                                            [conv_F2_numFilters],[1,2,2,1],[1,2,2,1])
        print("out_face_convF2: %s"%(out_face_convF2))
    with tf.variable_scope("conv_F3"):
        # Variables created here will be named "conv_E1/weights", "conv_E1/biases".
        out_face_convF3 = fn_conv_relu(out_face_convF2,
                                            [conv_F3_filterSize,conv_F3_filterSize,conv_F2_numFilters,conv_F3_numFilters],
                                            [conv_F3_numFilters])
        print("out_face_convF3: %s"%(out_face_convF3))

#Fully-Connected layers
    with tf.variable_scope("fc_E1_left"):
        #Connect left eye conv output to fully connected layer using reshape
        out_fc_E1_left_inputSize = 11*11*64
        out_fc_E1_left_temp = tf.reshape(out_left_convE3,[-1,out_fc_E1_left_inputSize])
        out_fc_E1_left = fn_fullyconn(out_fc_E1_left_temp, out_fc_E1_left_inputSize, fc_E1_numFilters)
        print("out_fc_E1_left: %s"%(out_fc_E1_left))
    with tf.variable_scope("fc_E1_right"):
        #Connect right eye conv output to fully connected layer using reshape
        out_fc_E1_right_inputSize = 11*11*64
        out_fc_E1_right_temp = tf.reshape(out_right_convE3,[-1,out_fc_E1_right_inputSize])
        out_fc_E1_right = fn_fullyconn(out_fc_E1_right_temp, out_fc_E1_right_inputSize, fc_E1_numFilters)
        print("out_fc_E1_right: %s"%(out_fc_E1_right))
    with tf.variable_scope("fc_E2"):
        #Concatenate left eye and right eye output to fully connected layer
        #out_fc_E2_temp = tf.concat(1,[out_fc_E1_left,out_fc_E1_right])#TF:0.8
        out_fc_E2_temp = tf.concat([out_fc_E1_left,out_fc_E1_right],1)#TF:1.0
        out_fc_E2 = fn_fullyconn(out_fc_E2_temp, 2*fc_E1_numFilters, fc_E2_numFilters)
        print("out_fc_E2: %s"%(out_fc_E2))

    with tf.variable_scope("fc_F1"):
        #Connect face conv output to fully connected layer using reshape
        out_fc_F1_inputSize = 11*11*64
        out_fc_F1_temp = tf.reshape(out_face_convF3,[-1,out_fc_F1_inputSize])
        out_fc_F1 = fn_fullyconn(out_fc_F1_temp, out_fc_F1_inputSize, fc_F1_numFilters)
        print("out_fc_F1: %s"%(out_fc_F1))
    with tf.variable_scope("fc_F2"):
        out_fc_F2 = fn_fullyconn(out_fc_F1, fc_F1_numFilters, fc_F2_numFilters)
        print("out_fc_F2: %s"%(out_fc_F2))
    
    with tf.variable_scope("fc_EFM1"):
        #Concatenate eyes and face output to fully connected layer
        #out_fc_EFM1_temp1 = tf.concat(1,[out_fc_E2,out_fc_F2])#TF:0.8
        out_fc_EFM1_temp1 = tf.concat([out_fc_E2,out_fc_F2],1)#TF:1.0
        out_fc_EFM1 = fn_fullyconn(out_fc_EFM1_temp1,
                                  fc_E2_numFilters+fc_F2_numFilters,
                                  fc_EFM1_numFilters)
        print("out_fc_EFM1: %s"%(out_fc_EFM1))
    with tf.variable_scope("fc_EFM2"):
        #Final layer with regression output
        out_fc_EFM2 = fn_fullyconn(out_fc_EFM1,fc_EFM1_numFilters,fc_EFM2_numFilters)
        print("out_fc_EFM2: %s"%(out_fc_EFM2))
    return out_fc_EFM2

#Outputs
predict_op = my_image_filter(eye_left,eye_right,face,face_mask)
error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(predict_op - y), 1)))
loss = tf.reduce_sum(tf.reduce_mean(tf.square(predict_op - y), 0))
optimization = tf.train.AdamOptimizer(lr).minimize(loss)

print("predict_op: %s"%predict_op)
print("error: %s"%error)
print("loss: %s"%loss)

print("Operations done")

# Create the collection.
tf.get_collection("validation_nodes")
# Add stuff to the collection.
tf.add_to_collection("validation_nodes", eye_left)
tf.add_to_collection("validation_nodes", eye_right)
tf.add_to_collection("validation_nodes", face)
tf.add_to_collection("validation_nodes", face_mask)
tf.add_to_collection("validation_nodes", predict_op)
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

#General parameters
batchSize = 100
batchSize_Test = 1000
numBatches = 10000#int(numSamplesTrain/batchSize)
maxEpochs = 1#30
logStep = 100
index_logStep = 0
timeTaken = 0

#Arrays for data collection
lossValue = np.zeros([maxEpochs*((numBatches//logStep)+1),1])
errorValue_Train = np.zeros([maxEpochs*((numBatches//logStep)+1),1])
errorValue_Test = np.zeros([maxEpochs*((numBatches//logStep)+1),1])

sess = tf.Session()
sess.run(init_op)
#Save graph
writer = tf.summary.FileWriter("./save_graph", sess.graph)

for epochID in range(maxEpochs):
    for batchID in range(numBatches):
        t1 = time.time()
		#Pick the batch for training
        ptr = np.random.choice(numSamplesTrain,size=batchSize,replace=False)#np.arange(batchSize)#np.arange(batchSize)#
        train_eye_left = train_eye_left_orig[ptr]/255.0
        train_eye_right = train_eye_right_orig[ptr]/255.0
        train_face = train_face_orig[ptr]/255.0
        train_face_mask = train_face_mask_orig[ptr]
        train_y = train_y_orig[ptr]
        #Create dictionary
        train_Dict = {eye_left:train_eye_left,eye_right:train_eye_right,face:train_face,face_mask:train_face_mask,y:train_y}
        #Optimize
        _,tempLoss,tempError = sess.run([optimization,loss,error],feed_dict=train_Dict)
        print("Opt(%d,%d) Loss: %g Error: %g in %g sec"%(epochID,batchID,tempLoss,tempError,time.time() - t1))
        timeTaken = timeTaken + time.time() - t1
        
        #Data collection
        if (batchID % logStep == 0) or (batchID == numBatches - 1):
            t2 = time.time()
            #Loss, Error for training data
            lossValue[index_logStep] = tempLoss
            errorValue_Train[index_logStep] = tempError
            #Error for testing data
            for idx in range(numSamplesTest//batchSize_Test):
                ptr = np.arange(idx*batchSize_Test,(idx+1)*batchSize_Test)
                val_eye_left = val_eye_left_orig[ptr]/255.0
                val_eye_right = val_eye_right_orig[ptr]/255.0
                val_face = val_face_orig[ptr]/255.0
                val_face_mask = val_face_mask_orig[ptr]
                val_y = val_y_orig[ptr]
                #Create dictionary
                val_Dict = {eye_left:val_eye_left,eye_right:val_eye_right,face:val_face,face_mask:val_face_mask,y:val_y}
                errorValue_Test[index_logStep] = errorValue_Test[index_logStep] + sess.run(error,feed_dict=val_Dict)

            errorValue_Test[index_logStep] = errorValue_Test[index_logStep]/(numSamplesTest/batchSize_Test)
            print("Log(%d,%d) in %g sec"%(epochID,batchID,time.time() - t2))
            timeTaken = timeTaken + time.time() - t2
            print("epochID: %d\tbatchID: %d\tErrorTrain: %g\tErrorTest: %g\tLoss: %g in %g sec"%(epochID,batchID,
                                                                                        errorValue_Train[index_logStep],
                                                                                        errorValue_Test[index_logStep],
                                                                                        lossValue[index_logStep],timeTaken))
            index_logStep = index_logStep + 1
            timeTaken = 0
        
#Save the model
save_path = saver.save(sess, "./save_model/my_model")
print("Save model finished")

#Save the plots
plt.figure(1)
plt.plot(lossValue[1:])
plt.xlabel('Iteration')
plt.ylabel('TrainingLoss')
titleString = 'Loss trend for training data with batch size: %d' %batchSize
plt.title(titleString)
plt.savefig('LossTrend.png')

plt.figure(2)
plt.plot(errorValue_Train[1:])
plt.xlabel('Iteration')
plt.ylabel('Error')
titleString = 'Error trend for random training data batch with batch size: %d' %batchSize
plt.title(titleString)
plt.savefig('Error-Train.png')

plt.figure(3)
plt.plot(errorValue_Test[1:])
plt.xlabel('Iteration')
plt.ylabel('Error')
titleString = 'Error trend for testing data'
plt.title(titleString)
plt.savefig('Error-Test.png')


