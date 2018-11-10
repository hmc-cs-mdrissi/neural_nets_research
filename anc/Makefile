generateArbitraryFor: GenerateArbitraryForTests.hs ArbitraryForTests.hs LambdaParser.hs ForParser.hs ForLambdaCommon.hs
	ghc GenerateArbitraryForTests.hs -main-is GenerateArbitraryForTests.main -o generateArbitraryFor

generateArbitraryCoffeeScript: ArbitraryCoffeeScriptTests.hs GenerateArbitraryCoffeeScriptTests.hs
	ghc GenerateArbitraryCoffeeScriptTests.hs -main-is GenerateArbitraryCoffeeScriptTests.main -o generateArbitraryCoffeeScript

clean:
	rm -f *.o *.hi generateArbitraryFor generateArbitraryCoffeeScript

cleanjson:
	rm -f *.json