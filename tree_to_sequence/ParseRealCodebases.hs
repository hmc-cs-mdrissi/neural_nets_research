module ParseRealCodebases where

import Data.Aeson
import Text.Parsec (parse)
import Text.Parsec.String (Parser)
import Text.Parsec.Char (satisfy)
import Text.JSON.Parsec (p_jvalue)
import Text.JSON.Types (JSValue)

import Data.Char (isSpace)
import Control.Applicative (many, some, liftA3)
import System.Environment (getArgs)
import System.IO (readFile)

lexeme :: Parser a -> Parser a
lexeme = (<* many (satisfy isSpace))

parseWord :: Parser String
parseWord = lexeme $ some $ satisfy (not . isSpace)

parsePrograms :: Parser [(String, String, JSValue)]
parsePrograms = some $ liftA3 (,,) parseWord parseWord (lexeme p_jvalue)

main :: IO ()
main = do [filename] <- getArgs
          programs <- readFile filename
          print $ parse parsePrograms filename programs
