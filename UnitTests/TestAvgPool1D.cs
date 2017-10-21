using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.Data2D;
using NNSharp.IO;
using NNSharp.Models;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestAvgPool1D
    {
        [TestMethod]
        public void Test_AvgPool1D_Execute()
        {
            // Initialize data.
            Data2D data = new Data2D(1, 3, 2, 1);
            data[0, 0, 0, 0] = 2;
            data[0, 1, 0, 0] = 1;
            data[0, 2, 0, 0] = 1;

            data[0, 0, 1, 0] = 1;
            data[0, 1, 1, 0] = 4;
            data[0, 2, 1, 0] = 2;

            AvgPool1DLayer pool = new AvgPool1DLayer(0, 1, 2);
            pool.SetInput(data);
            pool.Execute();
            Data2D output = pool.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 1.5, 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], 1.0, 0.0000001);

            Assert.AreEqual(output[0, 0, 1, 0], 2.5, 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], 3.0, 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_AvgPool1D_Null_Input()
        {
            Data2D data = null;
            AvgPool1DLayer pool = new AvgPool1DLayer(1, 1, 2);
            pool.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_AvgPool1D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            AvgPool1DLayer pool = new AvgPool1DLayer(1, 1, 2);
            pool.SetInput(data);
        }


        [TestMethod]
        public void Test_AvgPool1D_1_KerasModel()
        {
            string path = Resources.TestsFolder + "test_avgpool_1D_1_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 5, 2, 1);

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

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 2);
            Assert.AreEqual(ou.GetDimension().w, 3);

            Assert.AreEqual(ou[0, 0, 0, 0], 0.6666666865348816, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], 0.6666666865348816, 0.00001);
            Assert.AreEqual(ou[0, 1, 0, 0], 1.3333333730697632, 0.00001);
            Assert.AreEqual(ou[0, 1, 1, 0], 0.6666666865348816, 0.00001);
            Assert.AreEqual(ou[0, 2, 0, 0], 1.3333333730697632, 0.00001);
            Assert.AreEqual(ou[0, 2, 1, 0], 0.6666666865348816, 0.00001);
        }

        [TestMethod]
        public void Test_AvgPool1D_2_KerasModel()
        {
            string path = Resources.TestsFolder + "test_avgpool_1D_2_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 5, 2, 1);

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

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 2);
            Assert.AreEqual(ou.GetDimension().w, 2);

            Assert.AreEqual(ou[0, 0, 0, 0], 0.6666666865348816, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], 0.6666666865348816, 0.00001);
            Assert.AreEqual(ou[0, 1, 0, 0], 1.3333333730697632, 0.00001);
            Assert.AreEqual(ou[0, 1, 1, 0], 0.6666666865348816, 0.00001);
        }
    }
}
