using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using NNSharp.SequentialBased.SequentialLayers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace UnitTests
{
    [TestClass]
    public class TestSimpleRNN
    {
        [TestMethod]
        public void Test_SimpleRNN_Execute_Linear()
        {
            // Initialize data.
            Data2D data = new Data2D(1, 3, 3, 4);

            int l = 0;
            for (int b = 0; b < 4; ++b)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        l += 1;
                        data[0, w, c, b] = l % 5 + 1;
                    }
                }
            }

            // Initialize parameters.
            Data2D pms = new Data2D(1, 4, 4, 3);

            int k = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int u = 0; u < 4; ++u)
                {
                    k += 1;
                    pms[0, i, u, 0] = k % 5 - 2;
                }
            }

            k = 0;
            for (int i = 0; i < 4; ++i)
            {
                for (int u = 0; u < 4; ++u)
                {
                    k += 1;
                    pms[0, i, u, 1] = k % 5 - 2;
                }
            }

            pms[0, 0, 0, 2] = 1.0;
            pms[0, 0, 1, 2] = -1.0;
            pms[0, 0, 2, 2] = 2.0;
            pms[0, 0, 3, 2] = -4.0;


            SimpleRNNLayer rnn = new SimpleRNNLayer(4, 3, p => {  });
            rnn.SetWeights(pms);
            rnn.SetInput(data);
            rnn.Execute();
            Data2D output = rnn.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 4);
            Assert.AreEqual(dim.c, 4);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], -54, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 0], -39, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 0], 36, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 0], 72, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 1], 12, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 1], -19, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 1], -10, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 1], 10, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 2], -72, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 2], 16, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 2], 74, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 2], 68, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 3], -161, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 3], -14, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 3], 158, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 3], 141, 0.000001);
        }

        [TestMethod]
        public void Test_SimpleRNN_Execute_ReLu()
        {
            // Initialize data.
            Data2D data = new Data2D(1, 3, 3, 4);

            int l = 0;
            for (int b = 0; b < 4; ++b)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        l += 1;
                        data[0, w, c, b] = l % 5 + 1;
                    }
                }
            }

            // Initialize parameters.
            Data2D pms = new Data2D(1, 4, 4, 3);

            int k = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int u = 0; u < 4; ++u)
                {
                    k += 1;
                    pms[0, i, u, 0] = k % 5 - 2;
                }
            }

            k = 0;
            for (int i = 0; i < 4; ++i)
            {
                for (int u = 0; u < 4; ++u)
                {
                    k += 1;
                    pms[0, i, u, 1] = k % 5 - 2;
                }
            }

            pms[0, 0, 0, 2] = 1.0;
            pms[0, 0, 1, 2] = -1.0;
            pms[0, 0, 2, 2] = 2.0;
            pms[0, 0, 3, 2] = -4.0;


            SimpleRNNLayer rnn = new SimpleRNNLayer(4, 3, ReLuLayer.ReLuLambda );
            rnn.SetWeights(pms);
            rnn.SetInput(data);
            rnn.Execute();
            Data2D output = rnn.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 4);
            Assert.AreEqual(dim.c, 4);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 6, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 0], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 0], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 0], 0, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 1], 28, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 1], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 1], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 1], 0, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 2], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 2], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 2], 17, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 2], 34, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 3], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 3], 0, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 3], 25, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 3], 47, 0.000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_SimpleRNN_Null_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(1, 5, 5, 3);
            SimpleRNNLayer rnn = new SimpleRNNLayer(5, 3, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
            rnn.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_SimpleRNN_Null_Weights()
        {
            Data2D weights = null;
            SimpleRNNLayer rnn = new SimpleRNNLayer(5, 3, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_SimpleRNN_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(1, 5, 5, 3);
            SimpleRNNLayer rnn = new SimpleRNNLayer(5, 3, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
            rnn.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_SimpleRNN_WrongSizeinBatch_Weights()
        {
            Data2D weights = new Data2D(1, 5, 5, 4);
            SimpleRNNLayer rnn = new SimpleRNNLayer(5, 3, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_SimpleRNN_WrongSizeinChannel_Weights()
        {
            Data2D weights = new Data2D(1, 5, 7, 4);
            SimpleRNNLayer rnn = new SimpleRNNLayer(5, 3, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_SimpleRNN_WrongSizeinWidth_Weights()
        {
            Data2D weights = new Data2D(1, 3, 5, 4);
            SimpleRNNLayer rnn = new SimpleRNNLayer(5, 3, TanHLayer.TanHLambda);
            rnn.SetWeights(weights);
        }

        [TestMethod]
        public void Test_SimpleRNN_KerasModel()
        {
            string path = @"tests\test_simplernn_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            // Initialize data.
            Data2D data = new Data2D(1, 3, 3, 4);

            int l = 0;
            for (int b = 0; b < 4; ++b)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        l += 1;
                        data[0, w, c, b] = l % 5 + 1;
                    }
                }
            }

            Data2D output = model.ExecuteNetwork(data) as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 4);
            Assert.AreEqual(dim.c, 4);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], -54, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 0], -39, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 0], 36, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 0], 72, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 1], 12, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 1], -19, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 1], -10, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 1], 10, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 2], -72, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 2], 16, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 2], 74, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 2], 68, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 3], -161, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 3], -14, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 3], 158, 0.000001);
            Assert.AreEqual(output[0, 0, 3, 3], 141, 0.000001);
        }
    }
}
