using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;

namespace UnitTests
{
    [TestClass]
    public class TestMaxPool1D
    {
        [TestClass]
        public class TestMaxPool2D
        {
            [TestMethod]
            public void Test_MaxPool1D_Execute()
            {
                // Initialize data.
                Data2D data = new Data2D(1, 3, 2, 1);
                data[0, 0, 0, 0] = 1;
                data[0, 1, 0, 0] = 2;
                data[0, 2, 0, 0] = 0;

                data[0, 0, 1, 0] = 3;
                data[0, 1, 1, 0] = 4;
                data[0, 2, 1, 0] = 0;

                MaxPool1DLayer pool = new MaxPool1DLayer(0, 1, 2);
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
                Assert.AreEqual(output[0, 0, 0, 0], 2.0, 0.0000001);
                Assert.AreEqual(output[0, 1, 0, 0], 2.0, 0.0000001);

                Assert.AreEqual(output[0, 0, 1, 0], 4.0, 0.0000001);
                Assert.AreEqual(output[0, 1, 1, 0], 4.0, 0.0000001);
            }

            [TestMethod]
            [ExpectedException(typeof(System.Exception))]
            public void Test_MaxPool1D_Null_Input()
            {
                Data2D data = null;
                MaxPool1DLayer pool = new MaxPool1DLayer(1, 1, 2);
                pool.SetInput(data);
            }

            [TestMethod]
            [ExpectedException(typeof(System.Exception))]
            public void Test_MaxPool1D_DifferentData_Input()
            {
                DataArray data = new DataArray(5);
                MaxPool1DLayer pool = new MaxPool1DLayer(1, 1, 2);
                pool.SetInput(data);
            }
        }
    }
}
