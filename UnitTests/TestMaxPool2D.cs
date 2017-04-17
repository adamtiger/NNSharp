using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;

namespace UnitTests
{
    [TestClass]
    public class TestMaxPool2D
    {
        [TestMethod]
        public void Test_MaxPool2D_Execute()
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

            MaxPool2DLayer pool = new MaxPool2DLayer(0, 0, 1, 1, 2, 2);
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
            Assert.AreEqual(output[0, 0, 0, 0], 4.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 0, 0], 4.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], 4.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 0, 0], 4.0, 0.0000001);

            Assert.AreEqual(output[0, 0, 1, 0], 3.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 1, 0], 3.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], 1.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 1, 0], 1.0, 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_NullMaxPool_Input()
        {
            Data2D data = null;
            MaxPool2DLayer pool = new MaxPool2DLayer(1, 1, 1, 1, 2,2);
            pool.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            MaxPool2DLayer pool = new MaxPool2DLayer(1, 1, 1, 1, 2,2);
            pool.SetInput(data);
        }
    }
}
