# Build stage

FROM amazoncorretto:17 AS builder

WORKDIR /app

COPY . .

# gradlew 실행 권한 추가 (중요)
RUN chmod +x ./gradlew

# Gradle 빌드 (테스트 제외)
RUN ./gradlew clean build -x test

# Run stage

FROM amazoncorretto:17


WORKDIR /app

# builder에서 JAR 파일 복사
COPY --from=builder /app/build/libs/*SNAPSHOT.jar app.jar
ENTRYPOINT ["java", "-jar", "-Xms512m", "-Xmx512m", "app.jar"]