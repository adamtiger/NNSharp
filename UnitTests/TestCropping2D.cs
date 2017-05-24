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
    public class TestCropping2D
    {
        [TestMethod]
        public void Test_Cropping2D_Execute()
        {
            Data2D data = new Data2D(3, 4, 2, 1);

            int cntr = 0;

            for (int h = 0; h < 3; ++h)
            {
                for (int w = 0; w < 4; ++w)
                {
                    for (int c = 0; c < 2; ++c)
                    {
                        cntr += 1;
                        data[h, w, c, 0] = cntr;
                    }
                }
            }


            Cropping2DLayer crop = new Cropping2DLayer(1, 1, 2, 0);
            crop.SetInput(data);
            crop.Execute();
            Data2D output = crop.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 2);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], data[1, 2, 0, 0], 0.0000001);
            Assert.AreEqual(output[0, 0, 1, 0], data[1, 2, 1, 0], 0.0000001);
            Assert.AreEqual(output[0, 1, 0, 0], data[1, 3, 0, 0], 0.0000001);
            Assert.AreEqual(output[0, 1, 1, 0], data[1, 3, 1, 0], 0.0000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping2D_Null_Input()
        {
            Data2D data = null;
            Cropping2DLayer crop = new Cropping2DLayer(1, 2, 1, 1);
            crop.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping2D_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Cropping2DLayer crop = new Cropping2DLayer(2, 2, 1, 1);
            crop.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping2D_Negative_Trim()
        {
            Data2D data = new Data2D(8, 4, 3, 5);
            Cropping2DLayer crop = new Cropping2DLayer(4, -5, -1, 1);
            crop.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Cropping2D_TooMuchCropping_Trim()
        {
            Data2D data = new Data2D(4, 4, 3, 5);
            Cropping2DLayer crop = new Cropping2DLayer(2, 3, 1, 1);
            crop.SetInput(data);
        }

        [TestMethod]
        public void Test_Cropping2D_KerasModel()
        {
            string path = @"tests\test_crop_2D_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(4, 5, 2, 1);

            int l = 0;
            for (int h = 0; h < 4; ++h)
            {
                for (int w = 0; w < 5; ++w)
                {
                    l += 1;
                    inp[h, w, 0, 0] = l + 1;
                    inp[h, w, 1, 0] = -(l + 1);
                }
            }

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;

            Assert.AreEqual(ou.GetDimension().c, 2);
            Assert.AreEqual(ou.GetDimension().w, 2);
            Assert.AreEqual(ou.GetDimension().h, 2);

            Assert.AreEqual(ou[0, 0, 0, 0], 8.0, 0.00001);
            Assert.AreEqual(ou[0, 0, 1, 0], -8.0, 0.00001);
            Assert.AreEqual(ou[0, 1, 0, 0], 9.0, 0.00001);
            Assert.AreEqual(ou[0, 1, 1, 0], -9.0, 0.00001);

            Assert.AreEqual(ou[1, 0, 0, 0], 13.0, 0.00001);
            Assert.AreEqual(ou[1, 0, 1, 0], -13.0, 0.00001);
            Assert.AreEqual(ou[1, 1, 0, 0], 14.0, 0.00001);
            Assert.AreEqual(ou[1, 1, 1, 0], -14.0, 0.00001);
        }
    }
}
