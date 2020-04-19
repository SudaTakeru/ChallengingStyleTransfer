# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:52:25 2019

@author: sakicorp
"""

# 作成中０２１４

from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import threading
#from data_loader import DataLoader
from utility import *
from PIL import Image
from keras import backend as K
import tensorflow as tf
import argparse
import glob
from operator import itemgetter
from keras.engine.topology import Layer
from keras.engine import InputSpec

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu',type=str, default='1')
parser.add_argument('--lossfunction',type=str ,default='cycle_gan') 
parser.add_argument('--resultfolder',type=str,  default=r'train') 
parser.add_argument('--learn_time',  default='30000') 
parser.add_argument('--mode',  default='train') 

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu#"1"

class gan():
    def __init__(self,gen,psize,batch_size,lossfunc):
        # Input shape
        self.img_rows = psize[0]
        self.img_cols = psize[1]
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.batch_size = batch_size
        self.eps = 10**(-12)
        
        self.dataset_name = 'ct'
#        self.data_loader = DataLoader(dataset_name=self.dataset_name,
#                                      img_res=(self.img_rows, self.img_cols))
        
        self.data_loader = gen

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        
        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        
        self.depth = 7
        
        #optimizer = Adam(learning_rate=0.00001)
        goptimizer = Adam(learning_rate=0.00004)
        doptimizer = Adam(learning_rate=0.00001)

        # Build and compile the discriminator
        self.discriminator1 = self.build_discriminator()
        self.discriminator1.compile(loss='mse',
            optimizer=doptimizer,
            metrics=['accuracy'])
        
        self.discriminator2 = self.build_discriminator()
        self.discriminator2.compile(loss='mse',
            optimizer=doptimizer,
            metrics=['accuracy'])
        
        # Build the generator
        self.generator1 = self.build_generator()
        self.generator2 = self.build_generator()
#        self.generator._make_predict_function()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator1(img_B)
        fake_B = self.generator2(img_A)
        
        # Cycle Consistency 
        refake_A = self.generator1(fake_B)
        refake_B = self.generator2(fake_A)

        # For the combined model we will only train the generator
        self.discriminator1.trainable = False
#        self.discriminator._make_predict_function()

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator1([fake_A, img_A])
        valid2 = self.discriminator2([fake_B, img_B])
#        valid = Lambda(lambda x: K.log(x+self.eps))(valid)
        
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid,valid2,refake_A,refake_B])
        self.combined.compile(loss=['mse','mse',  lossfunc,  lossfunc],
                              loss_weights=[1,1,50,50],
                              optimizer=goptimizer)
        
        self.test = Model(inputs=[img_A, img_B], outputs=[fake_A,fake_B])
        self.test.compile(loss=[lossfunc, lossfunc],
                              loss_weights=[1, 1],
                              optimizer=goptimizer)
#        self.combined._make_predict_function()

    def build_generator(self):
#        depth = 4
        self.mode = 2
        self.features = 64
        self.k = 3
        
        input_layer=Input(self.img_shape)
        c1 = ReflectionPadding2D((4, 4))(input_layer)

        c1 = Conv2D(32, (9, 9), activation='linear', padding='valid', name='conv1')(c1)
        c1_b = BatchNormalization(axis=-1, name="batchnorm1")(c1)
        c1_b = Activation('relu')(c1_b)

        c2 = Conv2D(self.features, (self.k, self.k), activation='linear', padding='same', strides=(2, 2),
                           name='conv2')(c1_b)
        c2_b = BatchNormalization(axis=-1, name="batchnorm2")(c2)
        c2_b = Activation('relu')(c2_b)

        c3 = Conv2D(self.features, (self.k, self.k), activation='linear', padding='same', strides=(2, 2),
                           name='conv3')(c2_b)
        x = BatchNormalization(axis=-1, name="batchnorm3")(c3)
        x = Activation('relu')(x)
        
        r1 = self._residual_block(x, 1)
        r2 = self._residual_block(r1, 2)
        r3 = self._residual_block(r2, 3)
        r4 = self._residual_block(r3, 4)
        x = self._residual_block(r4, 5)
        
        x = UpSampling2D()(x)
        d3 = Conv2D(self.features, (self.k, self.k), activation="linear", padding="same",name="deconv3")(x)

        d3 = BatchNormalization(axis=-1, name="batchnorm4")(d3)
        d3 = Activation('relu')(d3)

        d3 = UpSampling2D()(d3)
        d2 = Conv2D(self.features, (self.k, self.k), activation="linear", padding="same",name="deconv2")(d3)

        d2 = BatchNormalization(axis=-1, name="batchnorm5")(d2)
        d2 = Activation('relu')(d2)

        d1 = ReflectionPadding2D((4, 4))(d2)
        d1 = Conv2D(1, (9, 9), activation='tanh', padding='valid', name='fastnet_conv')(d1)
        
        output_layer = Lambda(lambda x: (x+1)/2.0)(d1)
        
        return Model(input_layer,output_layer)
    
    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)
    
    def _residual_block(self, ip, id):
        init = ip

        x = ReflectionPadding2D((4, 4))(ip)
        x = Conv2D(128, (9,9), activation='linear', padding='valid',
                          name='res_conv_' + str(id) + '_1')(x)
        x = BatchNormalization(axis=-1, name="res_batchnorm_" + str(id) + "_1")(x)
        x = Activation('relu', name="res_activation_" + str(id) + "_1")(x)

        x = ReflectionPadding2D((4, 4))(x)
        x = Conv2D(self.features, (9,9), activation='linear', padding='valid',
                          name='res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=-1, name="res_batchnorm_" + str(id) + "_2")(x)

        m = Add()([x, init])#merge([x, init], mode='sum', name="res_merge_" + str(id))
        #m = Activation('relu', name="res_activation_" + str(id))(m)

        return m
    
    def train(self, epochs,testdata=None, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        
#        # pretraining
#        for i in range(10):
#            valid = np.ones((batch_size,) + self.disc_patch)+(0.3*(np.random.randn(batch_size,self.disc_patch[0],self.disc_patch[1],self.disc_patch[2])-0.5))
#            fake = np.zeros((batch_size,) + self.disc_patch)+0.3*np.random.randn(batch_size,self.disc_patch[0],self.disc_patch[1],self.disc_patch[2])
#            imgs_B, imgs_A = self.data_loader.__next__()
#            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        glosses = [0 for i in range(epochs)]
        dlosses = [0 for i in range(epochs)]
        for epoch in range(epochs):
            # Adversarial loss ground truths
            valid = np.ones((batch_size,) + self.disc_patch)#+(0.3*(np.random.randn(batch_size,self.disc_patch[0],self.disc_patch[1],self.disc_patch[2])))
            fake = np.zeros((batch_size,) + self.disc_patch)#+0.3*np.random.randn(batch_size,self.disc_patch[0],self.disc_patch[1],self.disc_patch[2])

#            print(threading.get_ident())
            batch_i = epoch+1
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            self.discriminator1.trainable = False
            self.discriminator2.trainable = False
            inp, outp = self.data_loader.__next__()
            
            imgs_A = inp[0]
            imgs_B = inp[1]
            
            fake_A = self.generator1.predict(imgs_B)
            fake_B = self.generator1.predict(imgs_A)
            
            # Train the discriminators (original images = real / generated = Fake)
            self.discriminator1.trainable = True
            self.discriminator2.trainable = True
            imgs_A1 = imgs_A #+ 0.01*np.random.randn(imgs_A.shape[0],imgs_A.shape[1],imgs_A.shape[2],imgs_A.shape[3])
            imgs_A2 = imgs_A + 0.01*np.random.randn(imgs_A.shape[0],imgs_A.shape[1],imgs_A.shape[2],imgs_A.shape[3])
            imgs_B1 = imgs_B #+ 0.01*np.random.randn(imgs_A.shape[0],imgs_A.shape[1],imgs_A.shape[2],imgs_A.shape[3])
            imgs_B2 = imgs_B + 0.01*np.random.randn(imgs_A.shape[0],imgs_A.shape[1],imgs_A.shape[2],imgs_A.shape[3])
            d_loss_real = self.discriminator1.train_on_batch([imgs_A1, imgs_A2], valid)
            d_loss_real2 = self.discriminator2.train_on_batch([imgs_B1, imgs_B2], valid)

            # -----------------
            #  Train Generator
            # -----------------
            
            for i in range(3):
                # Train the generators
                self.discriminator1.trainable = False
                self.discriminator2.trainable = False
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid,valid,imgs_A,imgs_B])
                
                inp, outp = self.data_loader.__next__()
                imgs_A = inp[0]
                imgs_B = inp[1]
                fake_A = self.generator1.predict(imgs_B)
                fake_B = self.generator2.predict(imgs_A)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.discriminator1.trainable = True
            self.discriminator2.trainable = True
            imgs_A1 = imgs_A + 0.01*np.random.randn(imgs_A.shape[0],imgs_A.shape[1],imgs_A.shape[2],imgs_A.shape[3])
            imgs_B1 = imgs_B + 0.01*np.random.randn(imgs_A.shape[0],imgs_A.shape[1],imgs_A.shape[2],imgs_A.shape[3])
            d_loss_fake = self.discriminator1.train_on_batch([fake_A, imgs_A1], fake)
            d_loss_fake2 = self.discriminator2.train_on_batch([fake_B, imgs_B1], fake)
            d_loss = np.add(d_loss_real, d_loss_fake)
            d_loss = np.add(d_loss,d_loss_real2)
            d_loss = np.add(d_loss, d_loss_fake2)
            d_loss = 0.25 * d_loss

            # -----------------
            #  Train Generator
            # -----------------

            for i in range(3):
                # Train the generators
                self.discriminator1.trainable = False
                self.discriminator2.trainable = False
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid,valid,imgs_A,imgs_B])
                
                inp, outp = self.data_loader.__next__()
                imgs_A = inp[0]
                imgs_B = inp[1]
                fake_A = self.generator1.predict(imgs_B)
                fake_B = self.generator1.predict(imgs_A)

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f, mse: %f] time: %s" % (epoch, epochs,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0], g_loss[1],
                                                                    elapsed_time))
            glosses[epoch] = g_loss[0]
            dlosses[epoch] = d_loss[0]
            
            # If at save interval => save generated image samples
            if batch_i % sample_interval == 0:
                self.sample_images(epoch,testdata, batch_i,sample_interval)
        plt.plot( range(epochs),glosses,label='generator loss')
        plt.plot( range(epochs),dlosses,label='discreminator loss')
        plt.legend()
        plt.savefig('loss')
                    
    def sample_images(self, epoch,testdata, batch_i=None,sample_interval=0):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        os.makedirs('weights/%s' % self.dataset_name, exist_ok=True)
#        r, c = 3, 3
        
        if testdata == None:
            inp, outp = self.data_loader.__next__()
            imgs_A = inp[0]
            imgs_B = inp[1]
        else:
            imgs_B = testdata[0]
            imgs_A = testdata[1]
        fake_A = self.generator1.predict(imgs_B)

#        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        imgs_B = 0.5 * imgs_B + 0.5
        imgs_A = 0.5 * imgs_A + 0.5
        fake_A = 0.5 * fake_A + 0.5
        
#        print(len(gen_imgs))
#        print(gen_imgs[0].shape)
        
        titles = ['Condition', 'Generated', 'Original']
        
        for i in range(fake_A.shape[3]):
            Image.fromarray(np.uint8(tone(fake_A[0,:,:,i],np.mean(imgs_A))*255)).save(r'images/'+self.dataset_name+r'/'+str(epoch)+'tone_'+str(i)+titles[1]+'.png')
            Image.fromarray(np.float32(fake_A[0,:,:,i])).save(r'images/'+self.dataset_name+r'/'+str(epoch)+titles[1]+'.tiff')
        
            if batch_i == sample_interval:
#                Image.fromarray(np.uint8(tone(imgs_B[0,:,:,i],np.mean(imgs_B))*255)).save(r'images/'+self.dataset_name+r'/'+'tone_'+str(i)+titles[0]+'.png')
#                Image.fromarray(np.float32(imgs_B[0,:,:,i])).save(r'images/'+self.dataset_name+r'/'+str(i)+titles[0]+'.tiff')
                Image.fromarray(np.uint8(tone(imgs_A[0,:,:,i],np.mean(imgs_A))*255)).save(r'images/'+self.dataset_name+r'/'+'tone_'+str(i)+titles[2]+'.png')
                Image.fromarray(np.float32(imgs_A[0,:,:,i])).save(r'images/'+self.dataset_name+r'/'+str(i)+titles[2]+'.tiff')
        self.combined.save_weights(r'weights/'+"gan_weights.hdf5") 
        self.generator1.save_weights(r'weights/'+"generator_weights.hdf5")                     
#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            cnt = 0
#            for j in range(c):
#                axs[i,j].imshow(gen_imgs[cnt][:,:,0])
#                axs[i, j].set_title(titles[i])
#                axs[i,j].axis('off')
#                cnt += 1
#        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
#        plt.close()

    def test(self,testdata=None):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        os.makedirs('weights/%s' % self.dataset_name, exist_ok=True)
#       
        start_time = datetime.datetime.now()            
        imgs_B = testdata[0]
        imgs_A = testdata[1]
        fake_A = self.generator1.predict(imgs_B)
#        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
#        imgs_B = 0.5 * imgs_B + 0.5
#        imgs_A = 0.5 * imgs_A + 0.5
#        fake_A = 0.5 * fake_A + 0.5
        
        
#        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A],axis=-1)
        
        elapsed_time = datetime.datetime.now() - start_time
        print('time:'+str(elapsed_time))
        titles = ['50shot', 'Generated', '600shot']
        for i in range(fake_A.shape[3]):
            Image.fromarray(np.uint8(tone(fake_A[0,:,:,i],np.mean(imgs_A))*255)).save(r'images/'+self.dataset_name+r'/test_'+'tone_'+titles[1]+'_'+str(i)+'.png')
            Image.fromarray(np.float32(fake_A[0,:,:,i])).save(r'images/'+self.dataset_name+r'/test_'+titles[1]+'_'+str(i)+'.tiff')
            Image.fromarray(np.uint8(tone(imgs_A[0,:,:,i],np.mean(imgs_A))*255)).save(r'images/'+self.dataset_name+r'/'+'tone_'+titles[2]+'_'+str(i)+'.png')
            Image.fromarray(np.float32(imgs_A[0,:,:,i])).save(r'images/'+self.dataset_name+r'/'+titles[2]+'_'+str(i)+'.tiff')
        for i in range(imgs_B.shape[3]):
            Image.fromarray(np.uint8(tone(imgs_B[0,:,:,i],np.mean(imgs_B))*255)).save(r'images/'+self.dataset_name+r'/'+'tone_'+titles[0]+'_'+str(i)+'.png')
            Image.fromarray(np.float32(imgs_B[0,:,:,i])).save(r'images/'+self.dataset_name+r'/'+titles[0]+'_'+str(i)+'.tiff')
            
#        mse0 = np.mean(np.square(gen_imgs[0]-gen_imgs[2]))
#        print('MSE_50_600:'+str(mse0))
#        
#        mse = np.mean(np.square(gen_imgs[1]-gen_imgs[2]))
#        print('MSE_gen_600:'+str(mse))
    
#%%
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
            
#%%
if __name__ == '__main__':
    
    path = r"C:\Users\sakicorp\Documents\suda"
    datapath1 = r"C:\Users\sakicorp\Documents\suda\20190722-DeepLearningReconstructionMaterial\Image\OK - with back comp\OffAxis\600shots"
    datapath2 = r"C:\Users\sakicorp\Documents\suda\20190722-DeepLearningReconstructionMaterial\Image\OK\OffAxis\600shots"
    resultpath = path+r"\python_script_sinogram\results_0130_gan"
    resultpath_raw = path+r"\python_script_sinogram\raws_model_0220_cyke_gan"

    mode = args.mode
    
    #%% parameter
    learning_batch = int(args.learn_time)#30000
    psize=[64,64] 
    batch_size = 8
    testpsize = [512,512]
    trim = [716,512]
    cliprange = [1500,4500]

    lossfunc = VL1E_HP3
        
    
    #%%
    path = datapath1 + r'\*.tif'
    path2 = datapath2 + r'\*.tif'
    filelist_with_BC = glob.glob(path);
    filelist2 = glob.glob(path2);
    
    sec300 = np.expand_dims(np.array(Image.open(filelist_with_BC[0])),-1)
    
    if not os.path.exists(resultpath):
       os.mkdir(resultpath)
    os.chdir(resultpath)
     
    gen =data_generator_gan(filelist2,filelist_with_BC,psize,batch_size,cliprange,learning_batch,trim)
    
    ti,to = gen.__next__()
    testdata=[ti[0],ti[1]]
    
    #%% training
    if 'train' == mode:
        gantrain = gan(gen,psize,batch_size,lossfunc)
        gantrain.train(epochs=learning_batch,testdata=testdata, batch_size=batch_size, sample_interval=500)
        gantrain.combined.save_weights("gan_weights.hdf5") 
        gantrain.generator.save_weights("generator_weights.hdf5")     
        gantrain.combined.save("gan_model.hdf5")
        gantrain.generator.save("generator_model.hdf5")
    
    #%% test
    if 'test' == mode:
        testgan =data_generator_gan(filelist_input,filelist_output,testpsize,1,cliprange,learning_batch,trim, magnification,sampling,test=True)
        ti,to = testgan.take_test_data()
        testdata=[ti,to]
        
        gantest = gan(gen,testpsize,batch_size,lossfunc,magnification,sampling)
        gantest.combined.load_weights("gan_weights.hdf5") 
        gantest.generator.load_weights("generator_weights.hdf5")     
    
        gantest.test(testdata=testdata)      

    #%% RAW 
    if 'raw' == mode:
        import time
        
        gantest = gan(gen,testpsize,batch_size,lossfunc,magnification,sampling)
        gantest.combined.load_weights("gan_weights.hdf5") 
        gantest.generator.load_weights("generator_weights.hdf5")     
        
        imgsize= [1944,1536]
        msize=[20,20]
        psize = testpsize
        
        wrpixels = imgsize[0] % (psize[0]-2*msize[0])
        wtime = 1+imgsize[0] // (psize[0]-2*msize[0])
        
        hrpixels = imgsize[1] % (psize[1]-2*msize[1])
        htime = 1+imgsize[1] // (psize[1]-2*msize[1])
        
        path = datapath1 + r'\*.tif'
        filelist = glob.glob(path);
        
        if not os.path.exists(resultpath_raw):
            os.mkdir(resultpath_raw)
        os.chdir(resultpath_raw)
        
        elapsed_time = np.zeros((len(filelist),))
        for filenum in range(len(filelist)-2):
            print(filelist[filenum])
                
            angle1 = np.expand_dims(np.array(Image.open(filelist[filenum])),-1)
            angle2 = np.expand_dims(np.array(Image.open(filelist[filenum+1])),-1)
            
            sec300 = np.concatenate([angle1,angle2],axis=-1)
            sec300 = (np.clip(np.float64(sec300),cliprange[0],cliprange[1])-cliprange[0])/(cliprange[1]-cliprange[0])
        
            elapsed_time2 = np.zeros((wtime*htime,))
            inputs = np.zeros((1,psize[0],psize[1],1))
            predicts = np.zeros((psize[0],psize[1],1,wtime*htime))
            avemask = np.zeros((imgsize[0],imgsize[1],1))
            start_time = time.process_time()         
                
            k=0
            for i in range(wtime):
                for ii in range(htime):
                    if i==0 and ii ==0:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[:psize[0],:psize[1],:],axis=0)
                        
                    elif i==0 and (ii+1)*psize[1]-2*ii*msize[1]<=imgsize[1]:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[:psize[0],ii*(psize[1]-2*msize[1]):(ii+1)*psize[1]-2*ii*msize[1],:],axis=0)
                        
                    elif (i+1)*psize[0]-2*i*msize[0]<=imgsize[0] and ii ==0:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[i*(psize[0]-2*msize[0]):(i+1)*psize[0]-2*i*msize[0],:psize[1],:],axis=0)
                        
                    elif i==0 and (ii+1)*psize[1]-2*ii*msize[1]>imgsize[1]:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[:psize[0],-psize[1]:,:],axis=0)
                        
                    elif (i+1)*psize[0]-2*i*msize[0]>imgsize[0] and ii==0:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[-psize[0]:,:psize[1],:],axis=0)
                        
                    elif (i+1)*psize[0]-2*i*msize[0]>imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]<=imgsize[1]:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[-psize[0]:,ii*(psize[1]-2*msize[1]):(ii+1)*psize[1]-2*ii*msize[1],:],axis=0)
                        
                    elif (i+1)*psize[0]-2*i*msize[0]<=imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]>imgsize[1]:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[i*(psize[0]-2*msize[0]):(i+1)*psize[0]-i*2*msize[0],-psize[1]:,:],axis=0)
                        
                    elif (i+1)*psize[0]-2*i*msize[0]>imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]>imgsize[1]:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[-psize[0]:,-psize[1]:,:],axis=0)
                        
                    elif (i+1)*psize[0]-2*i*msize[0]<=imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]<=imgsize[1]:
                        inputs[:,:,:,:,k] = np.expand_dims(sec300[i*(psize[0]-2*msize[0]):(i+1)*psize[0]-2*i*msize[0],ii*(psize[1]-2*msize[1]):(ii+1)*psize[1]-2*ii*msize[1],:],axis=0)
                        
                    inference_time = time.process_time()
                    predicts[:,:,:,k] = gantest.generator.predict(inputs[:,:,:,:,k])[0,:,:,:]
                    elapsed_time2[k] = time.process_time() - inference_time
                    print("{:.7f}sec".format(elapsed_time2[k]),flush=True)
                    k+=1
                    
            sumpredicts = np.zeros((imgsize[0],imgsize[1],1))
            k=0
            for i in range(wtime):
                for ii in range(htime):
                    if i==0 and ii ==0:
                        sumpredicts[:psize[0]-msize[0],:psize[1]-msize[1],:] = predicts[:-msize[0],:-msize[1],:,k] #+ sumpredicts[:psize[0]-msize[0],:psize[1]-msize[1]]
                        avemask[:psize[0]-msize[0],:psize[1]-msize[1]] = 1 + avemask[:psize[0]-msize[0],:psize[1]-msize[1]]
                    
                    elif i==0 and (ii+1)*psize[1]-2*ii*msize[1]<=imgsize[1]:
                        sumpredicts[:psize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1],:] = predicts[:-msize[0],msize[1]:-msize[1],:,k]# + sumpredicts[:psize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]]
                        avemask[:psize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]] = 1 + avemask[:psize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]]
                    
                    elif (i+1)*psize[0]-2*i*msize[0]<=imgsize[0] and ii ==0:
                        sumpredicts[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],:psize[1]-msize[1],:] = predicts[msize[0]:-msize[0],:-msize[1],:,k] #+ sumpredicts[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],:psize[1]-msize[1]]
                        avemask[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],:psize[1]-msize[1]] = 1 + avemask[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],:psize[1]-msize[1]]
                    
                    elif i==0 and (ii+1)*psize[1]-2*ii*msize[1]>imgsize[1]:
                        sumpredicts[:psize[0]-msize[0],-psize[1]+msize[1]:,:] = predicts[:-msize[0],msize[1]:,:,k] #+ sumpredicts[:psize[0]-msize[0],-psize[1]+msize[1]:]
                        avemask[:psize[0]-msize[0],-psize[1]+msize[1]:] = 1 + avemask[:psize[0]-msize[0],-psize[1]+msize[1]:]
                    
                    elif (i+1)*psize[0]-2*i*msize[0]>imgsize[0] and ii==0:
                        sumpredicts[-psize[0]+msize[0]:,:psize[1]-msize[1],:] = predicts[msize[0]:,:-msize[1],:,k] #+ sumpredicts[-psize[0]+msize[0]:,:psize[1]-msize[1]]
                        avemask[-psize[0]+msize[0]:,:psize[1]-msize[1]] = 1 + avemask[-psize[0]+msize[0]:,:psize[1]-msize[1]]
                                
                    elif (i+1)*psize[0]-2*i*msize[0]>imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]<=imgsize[1]:
                        sumpredicts[-psize[0]+msize[0]:,ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1],:] = predicts[msize[0]:,msize[1]:-msize[1],:,k] #+ sumpredicts[-psize[0]+msize[0]:,ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]]
                        avemask[-psize[0]+msize[0]:,ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]] = 1 + avemask[-psize[0]+msize[0]:,ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]]
                    
                    elif (i+1)*psize[0]-2*i*msize[0]<=imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]>imgsize[1]:
                        sumpredicts[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],-psize[1]+msize[1]:,:] = predicts[msize[0]:-msize[0],-psize[1]+msize[1]:,:,k]# + sumpredicts[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],-psize[1]+msize[1]:]
                        avemask[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],-psize[1]+msize[1]:] = 1 + avemask[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],-psize[1]+msize[1]:]
                    
                    elif (i+1)*psize[0]-2*i*msize[0]>imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]>imgsize[1]:
                        sumpredicts[-psize[0]+msize[0]:,-psize[1]+msize[1]:,:] = predicts[msize[0]:,msize[1]:,:,k] #+ sumpredicts[-psize[0]+msize[0]:,-psize[1]+msize[1]:]
                        avemask[-psize[0]+msize[0]:,-psize[1]+msize[1]:] = 1 + avemask[-psize[0]+msize[0]:,-psize[1]+msize[1]:]
                        
                    elif (i+1)*psize[0]-2*i*msize[0]<=imgsize[0] and (ii+1)*psize[1]-2*ii*msize[1]<=imgsize[1]:
                        sumpredicts[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1],:] = predicts[msize[0]:-msize[0],msize[1]:-msize[1],:,k] #+ sumpredicts[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]]
                        avemask[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]] = 1 + avemask[i*(psize[0]-2*msize[0])+msize[0]:(i+1)*psize[0]-2*i*msize[0]-msize[0],ii*(psize[1]-2*msize[1])+msize[1]:(ii+1)*psize[1]-2*ii*msize[1]-msize[1]]
                    
                    k+=1
                    
            avepredicts = sumpredicts#/avemask
            
            finalpredicts = avepredicts#np.uint16(avepredicts)*cliprange[1])
            
            finalpredicts = np.uint16(finalpredicts*(cliprange[1]-cliprange[0]) + cliprange[0])
            
            elapsed_time[filenum] = time.process_time() - start_time
            print("proess_time: {:.7f}sec".format(elapsed_time[k]),flush=True)
                   
            print("average_inference_time: {:.7f}sec".format(np.mean(elapsed_time2)))
            
            
            # save tif
            Image.fromarray(finalpredicts).save(filelist[filenum].split('\\')[-1])
        
            # save RAW file
            filename = filelist[filenum].split('\\')[-1].split(".")[0]
            dt = np.dtype('<u2')
            bytepredicts = finalpredicts.astype(dt)
            bytepredicts = bytepredicts.tobytes()
            finalpredicts.tofile(filename+str(".raw"))
            
        print("average_process_time: {:.7f}sec".format(np.mean(elapsed_time))) 
    