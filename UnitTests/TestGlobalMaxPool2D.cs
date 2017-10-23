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
    public class TestGlobalMaxPool2D
    {
        [TestMethod]
        public void Test_GlobalMaxPool2D_Execute()
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
            data[2, 1, 1, 0] = -1;

            data[0, 2, 1, 0] = -3;
            data[1, 2, 1, 0] = -1;
            data[2, 2, 1, 0] = 0;

            GlobalMaxPool2DLayer pool = new GlobalMaxPool2DLayer();
            pool.SetInput(data);
            pool.Execute();
            Data2D output = pool.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 4.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], 3.0, 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GlobalMaxPool2D_Null_Input()
        {
            Data2D data = null;
            GlobalMaxPool2DLayer pool = new GlobalMaxPool2DLayer();
            pool.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_GlobalMaxPool2D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            GlobalMaxPool2DLayer pool = new GlobalMaxPool2DLayer();
            pool.SetInput(data);
        }

        [TestMethod]
        public void Test_GlobalMaxPool2D_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_globalmaxpool_2D_model.json";
            string pathInput = Resources.TestsFolder + "test_globalmaxpool_2D_input.json";
            string pathOutput = Resources.TestsFolder + "test_globalmaxpool_2D_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }
    }
}
