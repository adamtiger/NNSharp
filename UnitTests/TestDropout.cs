using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.IO;
using NNSharp.DataTypes;
using NNSharp.Models;
using static NNSharp.DataTypes.Data2D;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestDropout
    {
        [TestMethod]
        public void Test_Dropout_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_dropout_model.json";
            string pathInput = Resources.TestsFolder + "test_dropout_input.json";
            string pathOutput = Resources.TestsFolder + "test_dropout_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }
    }
}