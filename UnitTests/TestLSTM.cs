using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;
using NNSharp.IO;
using NNSharp.Models;
using NNSharp.Kernels.CPUKernels;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestLSTM
    {
        [TestMethod]
        public void Test_LSTM_Execute_Linear()
        {
            // Initialize data.
            Data2D data = new Data2D(1, 3, 3, 5);

            int l = 0;
            for (int b = 0; b < 5; ++b)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        l += 1;
                        data[0, w, c, b] = (l % 5 + 1)/10.0;
                    }
                }
            }

            // Initialize parameters.
            Data2D pms = new Data2D(2, 3, 2, 12);

            int k = 0;
            int bc = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int u = 0; u < 8; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, bc] = (k % 5 - 2)/10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 4;
                    }
                }
            }

            k = 0;
            bc = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int u = 0; u < 8; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, 4 + bc] = (k % 5 - 2)/10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 4;
                    }
                }
            }

            pms[0, 0, 0, 8] = 1.0 / 10.0;
            pms[0, 0, 1, 8] = 2.0 / 10.0;
            pms[0, 0, 0, 9] = -1.0 / 10.0;
            pms[0, 0, 1, 9] = 0.0 / 10;
            pms[0, 0, 0, 10] = 3.0 / 10;
            pms[0, 0, 1, 10] = 4.0 / 10;
            pms[0, 0, 0, 11] = 5.0 / 10;
            pms[0, 0, 1, 11] = -2.0 / 10;

            LSTMLayer rnn = new LSTMLayer(2, 3, p => { }, p => { });
            rnn.SetWeights(pms);
            rnn.SetInput(data);
            rnn.Execute();
            Data2D output = rnn.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 5);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 0.01582, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 0], -0.00801, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 1], 0.01568, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 1], -0.00986, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 2], 0.01580, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 2], -0.011669, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 3], 0.00591, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 3], -0.009268, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 4], 0.01530, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 4], -0.010527, 0.00001);
        }

        [TestMethod]
        public void Test_LSTM_Execute_ReLu()
        {
            // Initialize data.
            Data2D data = new Data2D(1, 3, 3, 5);

            int l = 0;
            for (int b = 0; b < 5; ++b)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        l += 1;
                        data[0, w, c, b] = (l % 5 + 1) / 10.0;
                    }
                }
            }

            // Initialize parameters.
            Data2D pms = new Data2D(2, 3, 2, 12);

            int k = 0;
            int bc = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int u = 0; u < 8; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, bc] = (k % 5 - 2) / 10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 4;
                    }
                }
            }

            k = 0;
            bc = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int u = 0; u < 8; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, 4 + bc] = (k % 5 - 2) / 10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 4;
                    }
                }
            }

            pms[0, 0, 0, 8] = 1.0 / 10.0;
            pms[0, 0, 1, 8] = 2.0 / 10.0;
            pms[0, 0, 0, 9] = -1.0 / 10.0;
            pms[0, 0, 1, 9] = 0.0 / 10;
            pms[0, 0, 0, 10] = 3.0 / 10;
            pms[0, 0, 1, 10] = 4.0 / 10;
            pms[0, 0, 0, 11] = 5.0 / 10;
            pms[0, 0, 1, 11] = -2.0 / 10;


            LSTMLayer rnn = new LSTMLayer(2, 3, TanHKernel.TanHLambda, ReLuKernel.ReLuLambda);
            rnn.SetWeights(pms);
            rnn.SetInput(data);
            rnn.Execute();
            Data2D output = rnn.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 5);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 0.015777, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 0], 0.0, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 1], 0.01605, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 1], 0.0, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 2], 0.016398, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 2], 0.0, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 3], 0.006314, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 3], 0.0, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 4], 0.016303, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 4], 0.0, 0.00001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_LSTM_Null_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(1, 5, 5, 3);
            LSTMLayer rnn = new LSTMLayer(5, 3, TanHLayer.TanHLambda, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
            rnn.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_LSTM_Null_Weights()
        {
            Data2D weights = null;
            LSTMLayer rnn = new LSTMLayer(5, 3, TanHLayer.TanHLambda, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_LSTM_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(1, 5, 5, 3);
            LSTMLayer rnn = new LSTMLayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
            rnn.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_LSTM_WrongSizeinBatch_Weights()
        {
            Data2D weights = new Data2D(1, 5, 5, 4);
            LSTMLayer rnn = new LSTMLayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_LSTM_WrongSizeinChannel_Weights()
        {
            Data2D weights = new Data2D(1, 5, 7, 4);
            LSTMLayer rnn = new LSTMLayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_LSTM_WrongSizeinWidth_Weights()
        {
            Data2D weights = new Data2D(1, 3, 5, 4);
            LSTMLayer rnn = new LSTMLayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
        }

        [TestMethod]
        public void Test_LSTM_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_lstm_model.json";
            string pathInput = Resources.TestsFolder + "test_lstm_input.json";
            string pathOutput = Resources.TestsFolder + "test_lstm_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }
    }
}
