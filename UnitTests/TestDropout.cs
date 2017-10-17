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
        // Finish this !!!!
        [TestMethod]
        public void Test_Dropout_KerasModel()
        {
            string path = @"tests\test_dropout_model.json";
            var reader = new ReaderKerasModel(path);
            SequentialModel model = reader.GetSequentialExecutor();

            // Initialize data.
            Data2D data = new Data2D(1, 3, 3, 5);

            int l = 0;
            for (int b = 0; b < 5; ++b)
            {
                for (int w = 0; w < 3; ++w)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        l += 1;
                        data[0, w, c, b] = (l % 5 + 1) / 10.0;
                    }
                }
            }

            Data2D output = model.ExecuteNetwork(data) as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 5);
            Assert.AreEqual(dim.c, 2);
            Assert.AreEqual(dim.h, 1);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], 0.19632, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 0], 0.37259, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 1], 0.21991, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 1], 0.37473, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 2], 0.24834, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 2], 0.38176, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 3], 0.18727, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 3], 0.35267, 0.00001);

            Assert.AreEqual(output[0, 0, 0, 4], 0.166619, 0.00001);
            Assert.AreEqual(output[0, 0, 1, 4], 0.35275, 0.00001);
        }
    }
}