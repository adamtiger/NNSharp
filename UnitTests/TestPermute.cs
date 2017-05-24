using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;
using NNSharp.IO;
using NNSharp.Models;

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

        [TestMethod]
        public void Test_Permute_KerasModel()
        {
            string path = @"tests\test_permute_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D data = new Data2D(2, 3, 4, 1);

            int cntr = 0;
            for (int h = 0; h < 2; ++h)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 4; ++c)
                    {
                        cntr += 1;
                        data[h, w, c, 0] = cntr + 1;
                    }
                }
            }

            Data2D output = model.ExecuteNetwork(data) as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 3);
            Assert.AreEqual(dim.h, 4);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 2.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], 6.0, 0.0000001);
            Assert.AreEqual(output[0, 0, 2, 0], 10.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], 14.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], 18.0, 0.0000001);
            Assert.AreEqual(output[0, 1, 2, 0], 22.0, 0.0000001);

            Assert.AreEqual(output[1, 0, 0, 0], 3.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 1, 0], 7.0, 0.0000001);
            Assert.AreEqual(output[1, 0, 2, 0], 11.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 0, 0], 15.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 1, 0], 19.0, 0.0000001);
            Assert.AreEqual(output[1, 1, 2, 0], 23.0, 0.0000001);

            Assert.AreEqual(output[2, 0, 0, 0], 4.0, 0.0000001);
            Assert.AreEqual(output[2, 0, 1, 0], 8.0, 0.0000001);
            Assert.AreEqual(output[2, 0, 2, 0], 12.0, 0.0000001);
            Assert.AreEqual(output[2, 1, 0, 0], 16.0, 0.0000001);
            Assert.AreEqual(output[2, 1, 1, 0], 20.0, 0.0000001);
            Assert.AreEqual(output[2, 1, 2, 0], 24.0, 0.0000001);

            Assert.AreEqual(output[3, 0, 0, 0], 5.0, 0.0000001);
            Assert.AreEqual(output[3, 0, 1, 0], 9.0, 0.0000001);
            Assert.AreEqual(output[3, 0, 2, 0], 13.0, 0.0000001);
            Assert.AreEqual(output[3, 1, 0, 0], 17.0, 0.0000001);
            Assert.AreEqual(output[3, 1, 1, 0], 21.0, 0.0000001);
            Assert.AreEqual(output[3, 1, 2, 0], 25.0, 0.0000001);
        }
    }
}
