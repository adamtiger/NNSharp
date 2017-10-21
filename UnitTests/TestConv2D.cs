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
    public class TestConv2D
    {

        [TestMethod]
        public void Test_Conv2D_Execute()
        {
            // Initialize data.
            Data2D data = new Data2D(3, 3, 2, 1);
            data[0, 0, 0, 0] = 1;
            data[1, 0, 0, 0] = 2;
            data[2, 0, 0, 0] = 0;

            data[0, 1, 0, 0] = 3;
            data[1, 1, 0, 0] = 4;
            data[2, 1, 0, 0] = 0;

            data[0, 2, 0, 0] = 2;
            data[1, 2, 0, 0] = 2;
            data[2, 2, 0, 0] = 0;


            data[0, 0, 1, 0] = 0;
            data[1, 0, 1, 0] = 3;
            data[2, 0, 1, 0] = 1;

            data[0, 1, 1, 0] = 1;
            data[1, 1, 1, 0] = 1;
            data[2, 1, 1, 0] = 1;

            data[0, 2, 1, 0] = 3;
            data[1, 2, 1, 0] = 1;
            data[2, 2, 1, 0] = 0;

            // Initialize weights.
            Data2D weights = new Data2D(2, 2, 2, 1);
            weights[0, 0, 0, 0] = 1;
            weights[1, 0, 0, 0] = 2;

            weights[0, 1, 0, 0] = 2;
            weights[1, 1, 0, 0] = 3;


            weights[0, 0, 1, 0] = 1;
            weights[1, 0, 1, 0] = 1;

            weights[0, 1, 1, 0] = 1;
            weights[1, 1, 1, 0] = 3;

            Conv2DLayer conv = new Conv2DLayer(0, 0, 1, 1);
            conv.SetWeights(weights);
            conv.SetInput(data);
            conv.Execute();
            Data2D output = conv.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 1);
            Assert.AreEqual(dim.h, 2);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 30.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 0, 0], 18.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], 29.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 0, 0], 11.0, 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv2D_Null_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(3, 3, 3, 3);
            Conv2DLayer conv = new Conv2DLayer(1,1,1,1);
            conv.SetWeights(weights);
            conv.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv2D_Null_Weights()
        {
            Data2D weights = null;
            Conv2DLayer conv = new Conv2DLayer(1, 1, 1, 1);
            conv.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv2D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(3, 3, 3, 3);
            Conv2DLayer conv = new Conv2DLayer(1, 1, 1, 1);
            conv.SetWeights(weights);
            conv.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Conv2D_DifferentData_Weights()
        {
            DataArray weights = new DataArray(5);
            Conv2DLayer conv = new Conv2DLayer(1, 1, 1, 1);
            conv.SetWeights(weights);
        }

        [TestMethod]
        public void Test_Conv2D_1_KerasModel()
        {
            string path = Resources.TestsFolder + "test_conv_2D_1_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(4, 5, 2, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 1, 1, 0] = 1;
            inp[0, 2, 0, 0] = 0;
            inp[0, 2, 1, 0] = 0;
            inp[0, 3, 0, 0] = 2;
            inp[0, 3, 1, 0] = 1;
            inp[0, 4, 0, 0] = 2;
            inp[0, 4, 1, 0] = 1;


            inp[1, 0, 0, 0] = 0;
            inp[1, 0, 1, 0] = -1;
            inp[1, 1, 0, 0] = 1;
            inp[1, 1, 1, 0] = -2;
            inp[1, 2, 0, 0] = 3;
            inp[1, 2, 1, 0] = 1;
            inp[1, 3, 0, 0] = 2;
            inp[1, 3, 1, 0] = 0;
            inp[1, 4, 0, 0] = 2;
            inp[1, 4, 1, 0] = -3;


            inp[2, 0, 0, 0] = 1;
            inp[2, 0, 1, 0] = 2;
            inp[2, 1, 0, 0] = -2;
            inp[2, 1, 1, 0] = 0;
            inp[2, 2, 0, 0] = 3;
            inp[2, 2, 1, 0] = -3;
            inp[2, 3, 0, 0] = 2;
            inp[2, 3, 1, 0] = 1;
            inp[2, 4, 0, 0] = 2;
            inp[2, 4, 1, 0] = 0;


            inp[3, 0, 0, 0] = 1;
            inp[3, 0, 1, 0] = 2;
            inp[3, 1, 0, 0] = 0;
            inp[3, 1, 1, 0] = -2;
            inp[3, 2, 0, 0] = 3;
            inp[3, 2, 1, 0] = 1;
            inp[3, 3, 0, 0] = 2;
            inp[3, 3, 1, 0] = 3;
            inp[3, 4, 0, 0] = -3;
            inp[3, 4, 1, 0] = 1;

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 2);
            Assert.AreEqual(ou.GetDimension().w, 2);
            Assert.AreEqual(ou.GetDimension().h, 2);

            Assert.AreEqual(ou[0, 0, 0, 0], 19.0, 0.0001);
            Assert.AreEqual(ou[0, 0, 1, 0], -4.0, 0.0001);
            Assert.AreEqual(ou[0, 1, 0, 0], 11.5, 0.0001);
            Assert.AreEqual(ou[0, 1, 1, 0], 14.5, 0.0001);

            Assert.AreEqual(ou[1, 0, 0, 0], 12.5, 0.0001);
            Assert.AreEqual(ou[1, 0, 1, 0], 27.0, 0.0001);
            Assert.AreEqual(ou[1, 1, 0, 0], -24.5, 0.0001);
            Assert.AreEqual(ou[1, 1, 1, 0], 1.5, 0.0001);
        }

        [TestMethod]
        public void Test_Conv2D_2_KerasModel()
        {
            string path = Resources.TestsFolder + "test_conv_2D_2_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(4, 5, 2, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 1, 1, 0] = 1;
            inp[0, 2, 0, 0] = 0;
            inp[0, 2, 1, 0] = 0;
            inp[0, 3, 0, 0] = 2;
            inp[0, 3, 1, 0] = 1;
            inp[0, 4, 0, 0] = 2;
            inp[0, 4, 1, 0] = 1;


            inp[1, 0, 0, 0] = 0;
            inp[1, 0, 1, 0] = -1;
            inp[1, 1, 0, 0] = 1;
            inp[1, 1, 1, 0] = -2;
            inp[1, 2, 0, 0] = 3;
            inp[1, 2, 1, 0] = 1;
            inp[1, 3, 0, 0] = 2;
            inp[1, 3, 1, 0] = 0;
            inp[1, 4, 0, 0] = 2;
            inp[1, 4, 1, 0] = -3;


            inp[2, 0, 0, 0] = 1;
            inp[2, 0, 1, 0] = 2;
            inp[2, 1, 0, 0] = -2;
            inp[2, 1, 1, 0] = 0;
            inp[2, 2, 0, 0] = 3;
            inp[2, 2, 1, 0] = -3;
            inp[2, 3, 0, 0] = 2;
            inp[2, 3, 1, 0] = 1;
            inp[2, 4, 0, 0] = 2;
            inp[2, 4, 1, 0] = 0;


            inp[3, 0, 0, 0] = 1;
            inp[3, 0, 1, 0] = 2;
            inp[3, 1, 0, 0] = 0;
            inp[3, 1, 1, 0] = -2;
            inp[3, 2, 0, 0] = 3;
            inp[3, 2, 1, 0] = 1;
            inp[3, 3, 0, 0] = 2;
            inp[3, 3, 1, 0] = 3;
            inp[3, 4, 0, 0] = -3;
            inp[3, 4, 1, 0] = 1;

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 2);
            Assert.AreEqual(ou.GetDimension().w, 2);
            Assert.AreEqual(ou.GetDimension().h, 2);

            Assert.AreEqual(ou[0, 0, 0, 0], 2.0, 0.0001);
            Assert.AreEqual(ou[0, 0, 1, 0], 1.0, 0.0001);
            Assert.AreEqual(ou[0, 1, 0, 0], 7.5, 0.0001);
            Assert.AreEqual(ou[0, 1, 1, 0], 4.0, 0.0001);

            Assert.AreEqual(ou[1, 0, 0, 0], 19.5, 0.0001);
            Assert.AreEqual(ou[1, 0, 1, 0], 20.5, 0.0001);
            Assert.AreEqual(ou[1, 1, 0, 0], -19.5, 0.0001);
            Assert.AreEqual(ou[1, 1, 1, 0], -0.5, 0.0001);
        }
    }
}
