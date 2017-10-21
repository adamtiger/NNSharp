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
    public class TestAvgPool2D
    {
        [TestMethod]
        public void Test_AvgPool2D_Execute()
        {
            // Initialize data.
            Data2D data = new Data2D(3, 3, 2, 1);
            data[0, 0, 0, 0] = 2;
            data[1, 0, 0, 0] = 1;
            data[2, 0, 0, 0] = 1;

            data[0, 1, 0, 0] = 1;
            data[1, 1, 0, 0] = 4;
            data[2, 1, 0, 0] = 2;

            data[0, 2, 0, 0] = 1;
            data[1, 2, 0, 0] = 2;
            data[2, 2, 0, 0] = 0;


            data[0, 0, 1, 0] = 3;
            data[1, 0, 1, 0] = 6;
            data[2, 0, 1, 0] = 8;

            data[0, 1, 1, 0] = 5;
            data[1, 1, 1, 0] = 2;
            data[2, 1, 1, 0] = 4;

            data[0, 2, 1, 0] = 1;
            data[1, 2, 1, 0] = 4;
            data[2, 2, 1, 0] = 2;

            AvgPool2DLayer pool = new AvgPool2DLayer(0, 0, 1, 1, 2, 2);
            pool.SetInput(data);
            pool.Execute();
            Data2D output = pool.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 2);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 2.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 0, 0], 2.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], 2.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 0, 0], 2.0, 0.0000001);

            Assert.AreEqual(output[0, 0, 1, 0], 4.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 1, 0], 5.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], 3.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 1, 0], 3.0, 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_AvgPool2D_Null_Input()
        {
            Data2D data = null;
            AvgPool2DLayer pool = new AvgPool2DLayer(1, 1, 1, 1, 2, 2);
            pool.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_AvgPool2D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            AvgPool2DLayer pool = new AvgPool2DLayer(1, 1, 1, 1, 2, 2);
            pool.SetInput(data);
        }

        [TestMethod]
        public void Test_AvgPool2D_1_KerasModel()
        {
            string path = Resources.TestsFolder + "test_avgpool_2D_1_model.json";
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

            Assert.AreEqual(ou[0, 0, 0, 0], 1.1666666269302368, 0.0001);
            Assert.AreEqual(ou[0, 0, 1, 0], 0.0833333358168602, 0.0001);
            Assert.AreEqual(ou[0, 1, 0, 0], 1.5833333730697632, 0.0001);
            Assert.AreEqual(ou[0, 1, 1, 0], -0.25, 0.0001);

            Assert.AreEqual(ou[1, 0, 0, 0], 1.3333333730697632, 0.0001);
            Assert.AreEqual(ou[1, 0, 1, 0], 0.1666666716337204, 0.0001);
            Assert.AreEqual(ou[1, 1, 0, 0], 1.25, 0.0001);
            Assert.AreEqual(ou[1, 1, 1, 0], -0.25, 0.0001);
        }

        [TestMethod]
        public void Test_AvgPool2D_2_KerasModel()
        {
            string path = Resources.TestsFolder + "test_avgpool_2D_2_model.json";
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

            Assert.AreEqual(ou[0, 0, 0, 0], 1.1666666269302368, 0.0001);
            Assert.AreEqual(ou[0, 0, 1, 0], 0.0833333358168602, 0.0001);
            Assert.AreEqual(ou[0, 1, 0, 0], 1.5833333730697632, 0.0001);
            Assert.AreEqual(ou[0, 1, 1, 0], -0.25, 0.0001);

            Assert.AreEqual(ou[1, 0, 0, 0], 1.3333333730697632, 0.0001);
            Assert.AreEqual(ou[1, 0, 1, 0], 0.1666666716337204, 0.0001);
            Assert.AreEqual(ou[1, 1, 0, 0], 1.25, 0.0001);
            Assert.AreEqual(ou[1, 1, 1, 0], -0.25, 0.0001);
        }

    }
}
