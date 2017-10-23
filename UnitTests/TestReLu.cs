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
    public class TestReLu
    {

        [TestMethod]
        public void Test_ReLu_Execute()
        {
            relu = new ReLuLayer();

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

            Assert.AreEqual(output[0, 0, 0, 0], 4.0, 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], 2.0, 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], 0.0, 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], 3.0, 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], 0.0, 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], 0.0, 0.00000001);
        }

        [TestMethod]
        public void Test_ReLu_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_relu_model.json";
            string pathInput = Resources.TestsFolder + "test_relu_input.json";
            string pathOutput = Resources.TestsFolder + "test_relu_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }

        private ReLuLayer relu;
    }
}
