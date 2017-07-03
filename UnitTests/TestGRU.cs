using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.IO;
using static NNSharp.DataTypes.Data2D;
using NNSharp.Models;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.Kernels.CPUKernels;

namespace UnitTests
{
    [TestClass]
    public class TestGRU
    {
        [TestMethod]
        public void Test_GRU_Execute_Linear()
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
            Data2D pms = new Data2D(2, 3, 2, 9);

            int k = 0;
            int bc = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int u = 0; u < 6; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, bc] = (k % 5 - 2) / 10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 3;
                    }
                }
            }

            k = 0;
            bc = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int u = 0; u < 6; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, 3 + bc] = (k % 5 - 2) / 10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 3;
                    }
                }
            }

            pms[0, 0, 0, 6] = 1.0 / 10.0;
            pms[0, 0, 1, 6] = 2.0 / 10.0;
            pms[0, 0, 0, 7] = -1.0 / 10.0;
            pms[0, 0, 1, 7] = 0.0 / 10;
            pms[0, 0, 0, 8] = 3.0 / 10;
            pms[0, 0, 1, 8] = 4.0 / 10;


            GRULayer rnn = new GRULayer(2, 3, p => { }, p => { });
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
            Assert.AreEqual(output[0, 0, 0, 0], 0.20213, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 0], 0.39285, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 1], 0.22716, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 1], 0.39534, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 2], 0.25739, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 2], 0.40368, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 3], 0.18944, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 3], 0.36951, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 4], 0.16984, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 4], 0.36987, 0.00001);
        }

        [TestMethod]
        public void Test_GRU_Execute_ReLu()
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
            Data2D pms = new Data2D(2, 3, 2, 9);

            int k = 0;
            int bc = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int u = 0; u < 6; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, bc] = (k % 5 - 2) / 10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 3;
                    }
                }
            }

            k = 0;
            bc = 0;
            for (int i = 0; i < 2; ++i)
            {
                for (int u = 0; u < 6; ++u)
                {
                    k += 1;
                    pms[u % 2, i, 0, 3 + bc] = (k % 5 - 2) / 10.0;
                    if (k % 2 == 0)
                    {
                        bc += 1;
                        bc = bc % 3;
                    }
                }
            }

            pms[0, 0, 0, 6] = 1.0 / 10.0;
            pms[0, 0, 1, 6] = 2.0 / 10.0;
            pms[0, 0, 0, 7] = -1.0 / 10.0;
            pms[0, 0, 1, 7] = 0.0 / 10;
            pms[0, 0, 0, 8] = 3.0 / 10;
            pms[0, 0, 1, 8] = 4.0 / 10;

            GRULayer rnn = new GRULayer(2, 3, TanHKernel.TanHLambda, ReLuKernel.ReLuLambda);
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
            Assert.AreEqual(output[0, 0, 0, 0], 0.19632, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 0], 0.37259, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 1], 0.21991, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 1], 0.37473, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 2], 0.24834, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 2], 0.38176, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 3], 0.18727, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 3], 0.35267, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 4], 0.166619, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 4], 0.35275, 0.00001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GRU_Null_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(1, 5, 5, 3);
            GRULayer rnn = new GRULayer(5, 3, TanHLayer.TanHLambda, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
            rnn.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GRU_Null_Weights()
        {
            Data2D weights = null;
            GRULayer rnn = new GRULayer(5, 3, TanHLayer.TanHLambda, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GRU_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(1, 5, 5, 3);
            GRULayer rnn = new GRULayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
            rnn.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GRU_WrongSizeinBatch_Weights()
        {
            Data2D weights = new Data2D(1, 5, 5, 4);
            GRULayer rnn = new GRULayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GRU_WrongSizeinChannel_Weights()
        {
            Data2D weights = new Data2D(1, 5, 7, 4);
            GRULayer rnn = new GRULayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GRU_WrongSizeinWidth_Weights()
        {
            Data2D weights = new Data2D(1, 3, 5, 4);
            GRULayer rnn = new GRULayer(5, 3, TanHLayer.TanHLambda, p => { });
            rnn.SetWeights(weights);
        }

        [TestMethod]
        public void Test_GRU_KerasModel()
        {
            string path = @"tests\test_gru_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

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

            Data2D output = model.ExecuteNetwork(data) as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 5);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 0.19632, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 0], 0.37259, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 1], 0.21991, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 1], 0.37473, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 2], 0.24834, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 2], 0.38176, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 3], 0.18727, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 3], 0.35267, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 4], 0.166619, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 4], 0.35275, 0.00001);
        }
    }
}
