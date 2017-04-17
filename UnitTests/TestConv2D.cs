using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;

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
        public void Test_NullConv_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(3, 3, 3, 3);
            Conv2DLayer conv = new Conv2DLayer(1,1,1,1);
            conv.SetWeights(weights);
            conv.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullConv_Weights()
        {
            Data2D weights = null;
            Conv2DLayer conv = new Conv2DLayer(1, 1, 1, 1);
            conv.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(3, 3, 3, 3);
            Conv2DLayer conv = new Conv2DLayer(1, 1, 1, 1);
            conv.SetWeights(weights);
            conv.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Weights()
        {
            DataArray weights = new DataArray(5);
            Conv2DLayer conv = new Conv2DLayer(1, 1, 1, 1);
            conv.SetWeights(weights);
        }
    }
}
