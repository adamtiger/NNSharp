using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.Kernels;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.Data2D;

namespace UnitTests
{
    [TestClass]
    public class TestData2D
    {
        [TestInitialize]
        public void SetUp()
        {
            data = new Data2D(5, 4, 3, 1);
        }

        [TestMethod]
        public void Test_Data2D_ToZeros()
        {
            data.ToZeros();

            Assert.AreEqual(data[0, 0, 1, 0], 0.0, 0.00000001);
            Assert.AreEqual(data[1, 3, 1, 0], 0.0, 0.00000001);
            Assert.AreEqual(data[2, 2, 1, 0], 0.0, 0.00000001);
        }

        [TestMethod]
        public void Test_Data2D_ApplyToAll()
        {
            for (int b = 0; b < data.GetDimension().b; ++b)
            {
                for (int c = 0; c < data.GetDimension().c; ++c)
                {
                    for (int w = 0; w < data.GetDimension().w; ++w)
                    {
                        for (int h = 0; h < data.GetDimension().h; ++h)
                        {
                            data[h, w, c, b] = 0.0;
                        }
                    }
                }
            }

            data.ApplyToAll(p => { return p + 1; });

            Assert.AreEqual(data[1, 0, 1, 0], 1.0, 0.00000001);
            Assert.AreEqual(data[1, 3, 1, 0], 1.0, 0.00000001);
            Assert.AreEqual(data[3, 2, 0, 0], 1.0, 0.00000001);
        }

        [TestMethod]
        public void Test_Data2D_Cast()
        {
            DataArray array = new DataArray(5);
            array[0] = 1;
            array[1] = 2;
            array[2] = 3;
            array[3] = 0;
            array[4] = 7;

            Data2D casted = (Data2D)array;

            Assert.AreEqual(casted.GetDimension(), new Dimension(1,1,5,1));

            Assert.AreEqual(casted[0, 0, 0, 0], 1.0, 0.00000001);
            Assert.AreEqual(casted[0, 0, 1, 0], 2.0, 0.00000001);
            Assert.AreEqual(casted[0, 0, 2, 0], 3.0, 0.00000001);
            Assert.AreEqual(casted[0, 0, 3, 0], 0.0, 0.00000001);
            Assert.AreEqual(casted[0, 0, 4, 0], 7.0, 0.00000001);
        }

        private Data2D data;
    }
}
