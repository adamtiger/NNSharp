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
            string pathModel = Resources.TestsFolder + "test_conv_1D_1_model.json";
            string pathInput = Resources.TestsFolder + "test_conv_1D_1_input.json";
            string pathOutput = Resources.TestsFolder + "test_conv_1D_1_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }

        [TestMethod]
        public void Test_Conv1D_2_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_conv_1D_2_model.json";
            string pathInput = Resources.TestsFolder + "test_conv_1D_2_input.json";
            string pathOutput = Resources.TestsFolder + "test_conv_1D_2_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }
    }
}

