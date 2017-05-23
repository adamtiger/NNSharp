using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;

namespace UnitTests
{
    [TestClass]
    public class TestPermute
    {
        [TestMethod]
        public void Test_Permute_Execute()
        {
            Data2D data = new Data2D(2, 3, 4, 1);
            Data2D expected = new Data2D(4, 2, 3, 1);

            int cntr = 0;
            for (int h = 0; h < 2; ++h)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 4; ++c)
                    {
                        cntr += 1;
                        data[h, w, c, 0] = cntr;
                        expected[c, h, w, 0] = cntr;
                    }
                }
            }

            PermuteLayer perm = new PermuteLayer(3, 1, 2);
            perm.SetInput(data);
            perm.Execute();
            Data2D output = perm.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 3);
            Assert.AreEqual(dim.h, 4);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], expected[0, 0, 0, 0], 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], expected[0, 0, 1, 0], 0.0000001);
            Assert.AreEqual(output[0, 0, 2, 0], expected[0, 0, 2, 0], 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], expected[0, 1, 0, 0], 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], expected[0, 1, 1, 0], 0.0000001);
            Assert.AreEqual(output[0, 1, 2, 0], expected[0, 1, 2, 0], 0.0000001);

            Assert.AreEqual(output[1, 0, 0, 0], expected[1, 0, 0, 0], 0.0000001);
            Assert.AreEqual(output[1, 0, 1, 0], expected[1, 0, 1, 0], 0.0000001);
            Assert.AreEqual(output[1, 0, 2, 0], expected[1, 0, 2, 0], 0.0000001);
            Assert.AreEqual(output[1, 1, 0, 0], expected[1, 1, 0, 0], 0.0000001);
            Assert.AreEqual(output[1, 1, 1, 0], expected[1, 1, 1, 0], 0.0000001);
            Assert.AreEqual(output[1, 1, 2, 0], expected[1, 1, 2, 0], 0.0000001);

            Assert.AreEqual(output[2, 0, 0, 0], expected[2, 0, 0, 0], 0.0000001);
            Assert.AreEqual(output[2, 0, 1, 0], expected[2, 0, 1, 0], 0.0000001);
            Assert.AreEqual(output[2, 0, 2, 0], expected[2, 0, 2, 0], 0.0000001);
            Assert.AreEqual(output[2, 1, 0, 0], expected[2, 1, 0, 0], 0.0000001);
            Assert.AreEqual(output[2, 1, 1, 0], expected[2, 1, 1, 0], 0.0000001);
            Assert.AreEqual(output[2, 1, 2, 0], expected[2, 1, 2, 0], 0.0000001);

            Assert.AreEqual(output[3, 0, 0, 0], expected[3, 0, 0, 0], 0.0000001);
            Assert.AreEqual(output[3, 0, 1, 0], expected[3, 0, 1, 0], 0.0000001);
            Assert.AreEqual(output[3, 0, 2, 0], expected[3, 0, 2, 0], 0.0000001);
            Assert.AreEqual(output[3, 1, 0, 0], expected[3, 1, 0, 0], 0.0000001);
            Assert.AreEqual(output[3, 1, 1, 0], expected[3, 1, 1, 0], 0.0000001);
            Assert.AreEqual(output[3, 1, 2, 0], expected[3, 1, 2, 0], 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Permute_Null_Input()
        {
            Data2D data = null;
            PermuteLayer perm = new PermuteLayer(1,2,3);
            perm.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Permute_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            PermuteLayer perm = new PermuteLayer(1,2,3);
            perm.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Permute_WrongDimension1()
        {
            Data2D data = new Data2D(2, 4, 3, 5);
            PermuteLayer perm = new PermuteLayer(4, 2, 3);
            perm.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Permute_WrongDimension2()
        {
            Data2D data = new Data2D(2, 4, 3, 5);
            PermuteLayer perm = new PermuteLayer(1, 5, 3);
            perm.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Permute_WrongDimension3()
        {
            Data2D data = new Data2D(2, 4, 3, 5);
            PermuteLayer perm = new PermuteLayer(1, 3, -1);
            perm.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Permute_WrongDimension4()
        {
            Data2D data = new Data2D(2, 4, 3, 5);
            PermuteLayer perm = new PermuteLayer(1, 3, 3);
            perm.SetInput(data);
        }
    }
}
