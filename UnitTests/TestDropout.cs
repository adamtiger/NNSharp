using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.IO;
using NNSharp.DataTypes;
using NNSharp.Models;
using static NNSharp.DataTypes.Data2D;

namespace UnitTests
{
    [TestClass]
    public class TestDropout
    {
        [TestMethod]
        public void Test_Dropout_KerasModel()
        {
            string path = @"tests\test_dropout_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            // Initialize data.
            Data2D data = new Data2D(4, 4, 1, 1);

            data[0, 0, 0, 0] = 200;
            data[0, 1, 0, 0] = 139;
            data[0, 2, 0, 0] = 165;
            data[0, 3, 0, 0] = 98;

            data[1, 0, 0, 0] = 144;
            data[1, 1, 0, 0] = 21;
            data[1, 2, 0, 0] = 62;
            data[1, 3, 0, 0] = 62;

            data[2, 0, 0, 0] = 37;
            data[2, 1, 0, 0] = 59;
            data[2, 2, 0, 0] = 251;
            data[2, 3, 0, 0] = 169;

            data[3, 0, 0, 0] = 150;
            data[3, 1, 0, 0] = 233;
            data[3, 2, 0, 0] = 105;
            data[3, 3, 0, 0] = 47;


            Data2D output = model.ExecuteNetwork(data) as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 1);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], -11.5065536499, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 0], 18.6881141, 0.00001);
        }
    }
}