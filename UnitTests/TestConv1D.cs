using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;
using NNSharp.IO;
using NNSharp.Models;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestConv1D
    {
        [TestMethod]
        public void Test_Conv1D_Execute()
        {
            // Initialize data.
            Data2D data = new Data2D(1, 3, 2, 1);
            data[0, 0, 0, 0] = 1;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = 0;

            data[0, 0, 1, 0] = 3;
            data[0, 1, 1, 0] = 4;
            data[0, 2, 1, 0] = 0;

            // Initialize weights.
            Data2D weights = new Data2D(1, 2, 2, 1);
            weights[0, 0, 0, 0] = 1;
            weights[0, 1, 0, 0] = 2;

            weights[0, 0, 1, 0] = 2;
            weights[0, 1, 1, 0] = 3;

            Conv1DLayer conv = new Conv1DLayer(0, 1);
            conv.SetWeights(weights);
            conv.SetInput(data);
            conv.Execute();
            Data2D output = conv.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 1);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 23.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], 10.0, 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv1D_NullConv_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(1, 3, 3, 3);
            Conv1DLayer conv = new Conv1DLayer(1, 1);
            conv.SetWeights(weights);
            conv.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv1D_NullConv_Weights()
        {
            Data2D weights = null;
            Conv1DLayer conv = new Conv1DLayer(1, 1);
            conv.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv1D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(1, 3, 3, 3);
            Conv1DLayer conv = new Conv1DLayer(1, 1);
            conv.SetWeights(weights);
            conv.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv1D_DifferentData_Weights()
        {
            DataArray weights = new DataArray(5);
            Conv1DLayer conv = new Conv1DLayer(1, 1);
            conv.SetWeights(weights);
        }

        [TestMethod]
        public void Test_Conv1D_1_KerasModel()
        {
            string path = Resources.TestsFolder + "test_conv_1D_1_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 6, 4, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 0, 2, 0] = 2;
            inp[0, 0, 3, 0] = 1.5;

            inp[0, 1, 0, 0] = 1;
            inp[0, 1, 1, 0] = 0;
            inp[0, 1, 2, 0] = 0;
            inp[0, 1, 3, 0] = 0.6;

            inp[0, 2, 0, 0] = 2;
            inp[0, 2, 1, 0] = 1;
            inp[0, 2, 2, 0] = 2;
            inp[0, 2, 3, 0] = 2.5;

            inp[0, 3, 0, 0] = 1;
            inp[0, 3, 1, 0] = 0;
            inp[0, 3, 2, 0] = -1;
            inp[0, 3, 3, 0] = 0;

            inp[0, 4, 0, 0] = 1;
            inp[0, 4, 1, 0] = -2;
            inp[0, 4, 2, 0] = 3;
            inp[0, 4, 3, 0] = 3.5;

            inp[0, 5, 0, 0] = 2;
            inp[0, 5, 1, 0] = 1;
            inp[0, 5, 2, 0] = 4;
            inp[0, 5, 3, 0] = 3.5;

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 3);
            Assert.AreEqual(ou.GetDimension().w, 5);

            Assert.AreEqual(ou[0, 0, 0, 0], 9.399999618530273, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], -1.6999998092651367, 0.00001);
            Assert.AreEqual(ou[0, 0, 2, 0], 4.550000190734863, 0.00001);

            Assert.AreEqual(ou[0, 1, 0, 0], 8.100000381469727, 0.00001);
            Assert.AreEqual(ou[0, 1, 1, 0], 10.199999809265137, 0.00001);
            Assert.AreEqual(ou[0, 1, 2, 0], 2.75, 0.00001);

            Assert.AreEqual(ou[0, 2, 0, 0], 8.5, 0.00001);
            Assert.AreEqual(ou[0, 2, 1, 0], -4.0, 0.00001);
            Assert.AreEqual(ou[0, 2, 2, 0], 12.25, 0.00001);

            Assert.AreEqual(ou[0, 3, 0, 0], 0.0, 0.00001);
            Assert.AreEqual(ou[0, 3, 1, 0], 16.5, 0.00001);
            Assert.AreEqual(ou[0, 3, 2, 0], -6.25, 0.00001);

            Assert.AreEqual(ou[0, 4, 0, 0], 23.0, 0.00001);
            Assert.AreEqual(ou[0, 4, 1, 0], 7.5, 0.00001);
            Assert.AreEqual(ou[0, 4, 2, 0], 14.5, 0.00001);
        }

        [TestMethod]
        public void Test_Conv1D_2_KerasModel()
        {
            string path = Resources.TestsFolder + "test_conv_1D_2_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 6, 4, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 0, 2, 0] = 2;
            inp[0, 0, 3, 0] = 1.5;

            inp[0, 1, 0, 0] = 1;
            inp[0, 1, 1, 0] = 0;
            inp[0, 1, 2, 0] = 0;
            inp[0, 1, 3, 0] = 0.6;

            inp[0, 2, 0, 0] = 2;
            inp[0, 2, 1, 0] = 1;
            inp[0, 2, 2, 0] = 2;
            inp[0, 2, 3, 0] = 2.5;

            inp[0, 3, 0, 0] = 1;
            inp[0, 3, 1, 0] = 0;
            inp[0, 3, 2, 0] = -1;
            inp[0, 3, 3, 0] = 0;

            inp[0, 4, 0, 0] = 1;
            inp[0, 4, 1, 0] = -2;
            inp[0, 4, 2, 0] = 3;
            inp[0, 4, 3, 0] = 3.5;

            inp[0, 5, 0, 0] = 2;
            inp[0, 5, 1, 0] = 1;
            inp[0, 5, 2, 0] = 4;
            inp[0, 5, 3, 0] = 3.5;

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 3);
            Assert.AreEqual(ou.GetDimension().w, 3);

            Assert.AreEqual(ou[0, 0, 0, 0], 9.399999618530273, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], -1.6999998092651367, 0.00001);
            Assert.AreEqual(ou[0, 0, 2, 0], 4.550000190734863, 0.00001);

            Assert.AreEqual(ou[0, 1, 0, 0], 8.5, 0.00001);
            Assert.AreEqual(ou[0, 1, 1, 0], -4.0, 0.00001);
            Assert.AreEqual(ou[0, 1, 2, 0], 12.25, 0.00001);

            Assert.AreEqual(ou[0, 2, 0, 0], 23.0, 0.00001);
            Assert.AreEqual(ou[0, 2, 1, 0], 7.5, 0.00001);
            Assert.AreEqual(ou[0, 2, 2, 0], 14.5, 0.00001);
        }
    }
}

