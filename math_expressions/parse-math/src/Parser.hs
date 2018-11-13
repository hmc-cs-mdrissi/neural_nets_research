module Parser where

import Types

import Text.Parsec.Language (emptyDef)
import qualified Text.Parsec.Token as P
import Text.Parsec.Char
import Text.Parsec.Prim ((<|>), parse, getInput, try)
import Text.Parsec.Combinator
import Text.Parsec.String (Parser)
import Control.Applicative (some, many)
import Data.String
import Data.Function
import Data.Bifunctor
import Data.Maybe (fromMaybe)
import Text.Parsec.Error
import Debug.Trace (trace)
import Data.Aeson
import qualified Data.ByteString.Lazy as B
import Data.Text (Text)
import Data.Map (Map)
import qualified Data.Map.Lazy as Map

mathDef :: P.LanguageDef ()
mathDef = emptyDef {P.identStart = letter
                     ,P.identLetter = alphaNum <|> char '\''
                     ,P.opStart = oneOf "=.[];"
                     ,P.opLetter = oneOf "=+-"
                     ,P.reservedOpNames = ["\\div","\\times","_","^"]
                     ,P.reservedNames = []
                     }

integralP :: Parser ()
integralP = try $ string "\\int" *> whiteSpace

dxP :: Parser ()
dxP = try $ string "d x" *> whiteSpace

alphaP :: Parser ()
alphaP = try $ string "\\alpha" *> whiteSpace

betaP :: Parser ()
betaP = try $ string "\\beta" *> whiteSpace

gammaP :: Parser ()
gammaP = try $ string "\\gamma" *> whiteSpace

phiP :: Parser ()
phiP = try $ string "\\phi" *> whiteSpace

piP :: Parser ()
piP = try $ string "\\pi" *> whiteSpace

thetaP :: Parser ()
thetaP = try $ string "\\theta" *> whiteSpace

inftyP :: Parser ()
inftyP = try $ string "\\infty" *> whiteSpace

ldotsP :: Parser ()
ldotsP = try $ string "\\ldots" *> whiteSpace

lexer :: P.TokenParser ()
lexer = P.makeTokenParser mathDef

reserved :: String -> Parser ()
reserved = P.reserved lexer

equal :: Parser ()
equal = string "=" *> whiteSpace

factorial :: Parser ()
factorial = string "!" *> whiteSpace

plus :: Parser ()
plus = P.reservedOp lexer "+"

minus :: Parser ()
minus = P.reservedOp lexer "-"

divide :: Parser ()
divide = try $ P.reservedOp lexer "\\div"

mult :: Parser ()
mult = try $ P.reservedOp lexer "\\times"

marrow :: Parser ()
marrow = try $ string "\\rightarrow" *> whiteSpace

identifier :: Parser Char
identifier = letter

lexeme :: Parser a -> Parser a
lexeme = P.lexeme lexer

sinP :: Parser ()
sinP = try $ string "\\sin" *> whiteSpace

cosP :: Parser ()
cosP = try $ string "\\cos" *> whiteSpace

tanP :: Parser ()
tanP = try $ string "\\tan" *> whiteSpace

sqrtP :: Parser ()
sqrtP = try $ string "\\sqrt" *> whiteSpace

geqP :: Parser ()
geqP = try $ string "\\geq" *> whiteSpace

geP :: Parser ()
geP = string ">" *> whiteSpace

neqP :: Parser ()
neqP = try $ string "\\neq" *> whiteSpace

sumP :: Parser ()
sumP = try $ string "\\sum" *> whiteSpace

leqP :: Parser ()
leqP = try $ string "\\leq" *> whiteSpace

leP :: Parser ()
leP = string "<" *> whiteSpace

equalP :: Parser ()
equalP = string "=" *> whiteSpace

absP :: Parser ()
absP = string "|" *> whiteSpace

magP :: Parser ()
magP = string "||" *> whiteSpace

lParenP :: Parser ()
lParenP = string "(" *> whiteSpace

rParenP :: Parser ()
rParenP = string ")" *> whiteSpace

bigLParenP :: Parser ()
bigLParenP = try $ string "\\left(" *> whiteSpace

bigRParenP :: Parser ()
bigRParenP = try $ string "\\right)" *> whiteSpace

lBraceP :: Parser ()
lBraceP = char '{' *> whiteSpace

rBraceP :: Parser ()
rBraceP = char '}' *> whiteSpace

logP :: Parser()
logP = try $ string "\\log" *> whiteSpace

limP :: Parser()
limP = try $ string "\\lim" *> whiteSpace

fracP :: Parser()
fracP = try $ string "\\frac" *> whiteSpace

underscoreP :: Parser()
underscoreP = P.reservedOp lexer "_"

dollar :: Parser()
dollar = string "$" *> whiteSpace

pm :: Parser()
pm = try $ string "\\pm" *> whiteSpace

superscriptP :: Parser()
superscriptP = P.reservedOp lexer "^"

symbol :: Parser MathExpression
symbol = (do sym <- alphaP
             return $ Symbol Alpha)
          <|> 
          (do sym <- betaP
              return $ Symbol Beta)
          <|> 
          (do sym <- gammaP
              return $ Symbol Gamma)
          <|> 
          (do sym <- phiP
              return $ Symbol Phi)
          <|> 
          (do sym <- piP
              return $ Symbol Pi)
          <|> 
          (do sym <- thetaP
              return $ Symbol Theta)
          <|> 
          (do sym <- inftyP
              return $ Symbol Infty)
          <|> 
          (do sym <- ldotsP
              return $ Symbol Ldots)


varname :: Parser MathExpression
varname = do name <- identifier
             spaces
             return $ VarName name

whiteSpace :: Parser ()
whiteSpace = P.whiteSpace lexer

number :: Parser MathExpression
number = do num <- P.natural lexer 
            return $ IntegerM num


mathExpression' :: Int -> Bool -> Bool -> Bool -> Parser MathExpression
mathExpression' 0 bool sumBool absBool = (mathExpression' 1 bool sumBool absBool `chainl1` (pure (\x -> BinOp x ImplicitMult)))
mathExpression' 1 bool sumBool absBool = mathExpression' 2 bool sumBool absBool `chainl1` (
                        (leP *> pure (\x -> BinOp x Le)) <|> 
                        (leqP *> pure (\x -> BinOp x Leq)) <|> 
                        (geP *> pure (\x -> BinOp x Ge)) <|> 
                        (geqP *> pure (\x -> BinOp x Geq)) <|> 
                        (equalP *> pure (\x -> BinOp x Equal)) <|> 
                        (neqP *> pure (\x -> BinOp x Neq) )) -- all the comparison ops  :)
mathExpression' 2 bool sumBool absBool = mathExpression' 3 bool sumBool absBool `chainl1` ((marrow *> pure (\x -> BinOp x Marrow))) -- arrow :)
mathExpression' 3 bool sumBool absBool = mathExpression' 4 bool sumBool absBool `chainl1` ((plus *> pure (\x -> BinOp x Plus)) <|> (minus *> pure (\x -> BinOp x Minus)) <|> (pm *> pure (\x -> BinOp x BinaryPm))) -- plus, minus  :)
mathExpression' 4 bool sumBool absBool = mathExpression' 5 bool sumBool absBool `chainl1` ((mult *> pure (\x -> BinOp x Mult)) <|> (divide *> pure (\x -> BinOp x Div))) -- times, div :)
mathExpression' 5 bool sumBool absBool = do exp <- mathExpression' 6 bool sumBool absBool
                                            facs <- many factorial
                                            return $ iterate (flip PUnOp Factorial) exp !! (length facs) -- factorial
mathExpression' 6 bool sumBool absBool = do minusSigns <- many minus
                                            exp <- mathExpression' 65 bool sumBool absBool ----- 777
                                            return $ iterate (UnOp NegSign) exp !! (length minusSigns) -- negate
mathExpression' 65 bool sumBool absBool = (do pmSigns <- many pm
                                              exp <- mathExpression' 8 bool sumBool absBool ----- 777
                                              return $ iterate (UnOp UnaryPm) exp !! (length pmSigns))
                                          <|> 
                                          mathExpression' 8 bool sumBool absBool
                       

mathExpression' 7 bool sumBool absBool =
                    (do sinP
                        ((do superscriptP
                             exponent <- mathExpression' 7  False sumBool absBool
                             body <- mathExpression' 7 bool sumBool absBool
                             return $ UnOpExp Sin exponent body) <|>
                         (do body <- mathExpression' 7 bool sumBool absBool
                             return $ UnOpExp Sin Nil body)))
                <|> 
                    (do cosP 
                        ((do superscriptP
                             exponent <- mathExpression' 7  False sumBool absBool
                             body <- mathExpression' 7 bool sumBool absBool
                             return $ UnOpExp Cos exponent body) <|>
                         (do body <- mathExpression' 7 bool sumBool absBool
                             return $ UnOpExp Cos Nil body)))
                <|> 
                    (do tanP
                        ((do superscriptP
                             exponent <- mathExpression' 7  False sumBool absBool
                             body <- mathExpression' 7 bool sumBool absBool
                             return $ UnOpExp Tan exponent body) <|>
                         (do body <- mathExpression' 7 bool sumBool absBool
                             return $ UnOpExp Tan Nil body)))
                <|> 
                    (do sqrtP
                        body <- mathExpression' 7 bool sumBool absBool
                        return $ UnOp Sqrt body)
                <|>
                    (do logP
                        ((do underscoreP
                             logB <- mathExpression' 7  False sumBool absBool --TODO: CHANGE THESE BACK!!!
                             logV <- mathExpression' 7 False sumBool absBool 
                             return $ DoubOp LogOp logB logV) <|>
                         (do logV <- mathExpression' 7 False sumBool absBool 
                             return $ DoubOp LogOp Nil logV)))
                <|> 
                   (do limP
                       underscoreP
                       limB <- mathExpression' 7 False sumBool absBool
                       limV <- mathExpression' 7 True sumBool absBool
                       return $ DoubOp LimOp limB limV)
                <|> 
                   (do fracP
                       numerator <- mathExpression' 7 False sumBool absBool
                       denominator <- mathExpression' 7 False sumBool absBool
                       return $ DoubOp FracOp numerator denominator)
                <|> 
                   try (do sumP
                           lowerBound <- optionMaybe (do underscoreP
                                                         mathExpression' 7 bool True absBool)
                           upperBound <- optionMaybe (do superscriptP
                                                         mathExpression' 7 bool True absBool)
                           bodyValue <- mathExpression' 0 bool sumBool absBool
                           let lowerBoundValue = fromMaybe Nil lowerBound
                           let upperBoundValue = fromMaybe Nil upperBound
                           return $ Sum lowerBoundValue upperBoundValue bodyValue)
                <|> 
                   try (do sumP
                           upperBound <- optionMaybe (do superscriptP
                                                         mathExpression' 7 bool True absBool)
                           lowerBound <- optionMaybe (do underscoreP
                                                         mathExpression' 7 bool True absBool)
                           bodyValue <- mathExpression' 0 bool sumBool absBool
                           let lowerBoundValue = fromMaybe Nil lowerBound
                           let upperBoundValue = fromMaybe Nil upperBound
                           return $ Sum lowerBoundValue upperBoundValue bodyValue)
                <|>
                    (do integralP
                        body <- mathExpression' 7 False sumBool absBool 
                        dxP 
                        return $ Integral Nil Nil body)
                <|>
                  mathExpression' 9 bool sumBool absBool ---888

mathExpression' 8 bool sumBool absBool =  if sumBool then  --- 999
                                            (mathExpression' 7 False sumBool absBool `chainl1` (underscoreP *> pure (\x -> BinOp x SubscriptOp))) else 
                                            (mathExpression' 7 bool sumBool absBool `chainl1` (((underscoreP *> pure (\x -> BinOp x SubscriptOp))) <|> 
                                                                                                (trace (show bool) superscriptP *> pure (\x -> BinOp x SuperscriptOp))))

-- subscript, superscript  :)             
mathExpression' 9 bool sumBool absBool = (if (bool && not sumBool) then ((varname <|> number <|> symbol) `chainl1` (pure (\x -> BinOp x ImplicitMult))) else (varname <|> number <|> symbol))
                    <|> 
                       case absBool of 
                         True -> fail "nothing is working"
                         False -> (do
                                      magP
                                      expr <- mathExpression' 0 True False True
                                      magP
                                      return $ Container Magnitude expr Magnitude)
                    <|> 
                       case absBool of 
                         True -> fail "oh no"
                         False -> (do
                                      absP
                                      expr <- mathExpression' 0 True False True
                                      trace "final absp" absP
                                      return $ Container AbsBar expr AbsBar)
                   <|>
                       (do lParenP
                           expr <- mathExpression' 0 True False False
                           inputLeft <- getInput
                           rParenP
                           return $ Container LeftParen expr RightParen)
                    <|>
                       (do bigLParenP
                           expr <- mathExpression' 0 True False False
                           inputLeft <- getInput
                           bigRParenP
                           return $ Container LeftParen expr RightParen)
                    <|> 
                       (do lBraceP
                           expr <- mathExpression' 0 True False False
                           inputLeft <- getInput
                           rBraceP
                           return $ Container LeftBrace expr RightBrace)

parseMath :: String -> Either ParseError MathExpression
parseMath = parse (optional dollar *> mathExpression' 0 True False False) ""

