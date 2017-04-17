using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;

namespace UnitTests
{
    [TestClass]
    public class TestBias2D
    {
        [TestMethod]
        public void Test_Bias2D_Execute()
        {
            Data2D data = new Data2D(2, 3, 2, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 5;
            data[0, 2, 0, 0] = -2;
            data[1, 0, 0, 0] = 6;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            data[0, 0, 1, 0] = 1;
            data[0, 1, 1, 0] = 2;
            data[0, 2, 1, 0] = 3;
            data[1, 0, 1, 0] = 0;
            data[1, 1, 1, 0] = 9;
            data[1, 2, 1, 0] = -3;

            DataArray biases = new DataArray(2);
            biases[0] = 1.5;
            biases[1] = 2.0;

            Bias2DLayer bias = new Bias2DLayer();
            bias.SetWeights(biases);
            bias.SetInput(data);
            bias.Execute();
            Data2D output = bias.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], 5.5, 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], 6.5, 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], -0.5, 0.00000001);
            Assert.AreEqual(output[1, 0, 0, 0], 7.5, 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], 0.5, 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], -1.5, 0.00000001);

            Assert.AreEqual(output[0, 0, 1, 0], 3.0, 0.00000001);
            Assert.AreEqual(output[0, 1, 1, 0], 4.0, 0.00000001);
            Assert.AreEqual(output[0, 2, 1, 0], 5.0, 0.00000001);
            Assert.AreEqual(output[1, 0, 1, 0], 2.0, 0.00000001);
            Assert.AreEqual(output[1, 1, 1, 0], 11.0, 0.00000001);
            Assert.AreEqual(output[1, 2, 1, 0], -1.0, 0.00000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullData()
        {
            Data2D data = null;

            DataArray biases = new DataArray(2);
            biases[0] = 1.5;
            biases[1] = 2.0;

            Bias2DLayer bias = new Bias2DLayer();
            bias.SetWeights(biases);
            bias.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullBias()
        {
            DataArray biases = null;
            Bias2DLayer bias = new Bias2DLayer();
            bias.SetWeights(biases);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Bias()
        {
            Data2D biases = null;
            Bias2DLayer bias = new Bias2DLayer();
            bias.SetWeights(biases);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Input()
        {
            DataArray data = new DataArray(5);

            DataArray biases = new DataArray(2);
            biases[0] = 1.5;
            biases[1] = 2.0;

            Bias2DLayer bias = new Bias2DLayer();
            bias.SetWeights(biases);
            bias.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentSizes()
        {
            Data2D data = new Data2D(4, 5, 1, 5);

            DataArray biases = new DataArray(2);
            biases[0] = 1.5;
            biases[1] = 2.0;

            Bias2DLayer bias = new Bias2DLayer();
            bias.SetWeights(biases);
            bias.SetInput(data);
        }
    }
}
