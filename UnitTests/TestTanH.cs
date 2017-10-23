using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestTanH
    {
        [TestMethod]
        public void Test_TanH_Execute()
        {
            relu = new TanHLayer();

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            relu.SetInput(data);

            relu.Execute();

            Data2D output = relu.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], tanh(4.0), 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], tanh(2.0), 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], tanh(-2.0), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], tanh(3.0), 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], tanh(-1.0), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], tanh(-3.0), 0.00000001);
        }

        private double tanh(double x)
        {
            return 2.0 / (1 + Math.Exp(-2 * x)) - 1.0;
        }

        [TestMethod]
        public void Test_TanH_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_tanh_model.json";
            string pathInput = Resources.TestsFolder + "test_tanh_input.json";
            string pathOutput = Resources.TestsFolder + "test_tanh_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }

        private TanHLayer relu;
    }
}
